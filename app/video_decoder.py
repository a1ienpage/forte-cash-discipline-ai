"""
Модуль декодирования видео и построения точной временной шкалы на базе PyAV (`av`) (Раздел 6 ТЗ).

Обеспечивает:
1. Декодирование видеопотока через PyAV (`av.open`) с вычислением времени каждого кадра строго по
   `frame.pts * stream.time_base` относительно нормализованного начала потока (`start_offset_sec`).
2. Корректную работу с переменной частотой кадров (VFR) и ненулевым начальным смещением потока (`start_pts`).
   Запрещено полагаться только на `frame_index / fps`.
3. Строгий реестр декодированных кадров (`DecodedFrame`) с уникальными `frame_id`, фактическими `pts` и `timestamp_sec`.
4. Двухэтапную выборку кадров:
   - Первичный обзорный проход по всей длительности видео `[0, duration_sec]` с задокументированным `max_interval_sec`.
   - Плотный уточняющий проход (`sample_dense_window`) в окрестности кандидата события с контекстом до и после.
5. Кодирование и техническую верификацию доказательных кадров (JPEG + base64) по реально декодированным `frame_id`.
"""

import base64
import io
import math
import os
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Optional, Tuple, Any

import av
import cv2
import numpy as np
from PIL import Image, ImageDraw


class VideoDecodeError(RuntimeError):
    """Явная техническая ошибка декодирования видеоконтейнера или видеопотока (переводит задачу в failed)."""


@dataclass
class DecodedFrame:
    frame_id: str
    decode_index: int
    pts: int
    raw_pts_sec: float
    timestamp_sec: float
    width: int
    height: int
    bgr: np.ndarray

    def to_pil(self) -> Image.Image:
        rgb = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)


@dataclass
class VideoTimelineBundle:
    video_path: str
    codec_name: str
    width: int
    height: int
    nominal_fps: float
    time_base_str: str
    time_base_float: float
    start_offset_sec: float
    offset_normalization_rule: str
    duration_sec: float
    is_vfr_detected: bool
    max_observation_interval_sec: float
    short_action_limitation_note: str
    coarse_frames: List[DecodedFrame]
    all_decoded_frames: Dict[str, DecodedFrame]
    frame_pts_map: Dict[str, Dict[str, Any]] = None
    pts_fallback_used: bool = False
    missing_pts_frames_count: int = 0

    def __post_init__(self) -> None:
        if self.frame_pts_map is None:
            self.frame_pts_map = {
                fid: {
                    "frame_id": fid,
                    "decode_index": df.decode_index,
                    "pts": df.pts,
                    "raw_pts_sec": df.raw_pts_sec,
                    "timestamp_sec": df.timestamp_sec,
                }
                for fid, df in (self.all_decoded_frames or {}).items()
            }


def _downscale_bgr(bgr: np.ndarray, max_dim: int = 640) -> np.ndarray:
    """Уменьшает BGR-кадр для обзорного буфера с сохранением пропорций, ограничивая расход памяти (F05)."""
    h, w = bgr.shape[:2]
    if max(h, w) <= max_dim:
        return bgr
    scale = float(max_dim) / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


class PyAVVideoDecoder:
    """
    Потоковый декодер видеопотока с ограниченным потреблением памяти (F05),
    поддержкой PTS/time_base, VFR и двухпроходного извлечения кадров по требованию.
    """

    @staticmethod
    def probe_container_metadata(video_path: str) -> Dict[str, Any]:
        """
        Быстрое чтение метаданных контейнера и заголовка видеопотока без декодирования всех кадров (F14).
        Используется в GET /api/v1/dataset для мгновенного ответа без блокировки CPU/RAM.
        """
        if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
            raise VideoDecodeError(f"Файл видео не существует или пуст: {video_path}")
        try:
            container = av.open(video_path)
        except Exception as e:
            raise VideoDecodeError(f"Нечитаемый видеоконтейнер ({os.path.basename(video_path)}): {e}") from e
        try:
            if not container.streams.video:
                raise VideoDecodeError(f"В контейнере отсутствует видеопоток: {video_path}")
            stream = container.streams.video[0]
            w = int(stream.codec_context.width or stream.width or 0)
            h = int(stream.codec_context.height or stream.height or 0)
            avg_rate = float(stream.average_rate) if stream.average_rate else 30.0
            tb_fraction = stream.time_base
            dur = 0.0
            if stream.duration is not None and tb_fraction is not None:
                dur = float(stream.duration * tb_fraction)
            elif container.duration is not None:
                dur = float(container.duration) / float(av.time_base)
            return {
                "width": w,
                "height": h,
                "resolution": f"{w}x{h}",
                "fps": round(avg_rate, 3),
                "duration_sec": round(max(0.0, dur), 2),
                "codec_name": stream.codec_context.name if stream.codec_context else "unknown",
            }
        finally:
            container.close()

    @staticmethod
    def _reconstruct_missing_pts(
        raw_pts: List[Optional[int]],
        tb_float: float,
        nominal_fps: float,
        stream_start_pts: Optional[int],
    ) -> List[float]:
        """
        Единая реконструкция временной шкалы при отсутствующих `frame.pts` (R08; ревью 0909, п. 4.4).
        Используется И при построении реестра (`decode_timeline`), И при повторном извлечении кадров
        (`decode_frames_by_ids`), поэтому одно и то же восстановленное время кадра совпадает на всех этапах.
        - Внутренний пропуск: линейная интерполяция между ОБЕИМИ соседними известными метками
          (результат строго между соседями — обратный ход времени исключён при монотонных известных PTS).
        - Начальный пропуск: интерполяция от `stream.start_time` (кадр #0) до первой известной метки, либо, если
          `start_time` неизвестен/не меньше первой метки, обратная экстраполяция с номинальным шагом `1/fps`.
        - Конечный пропуск: экстраполяция с шагом последних двух известных меток (или `1/fps`).
        - Полное отсутствие PTS: `start_time + i/fps`.
        Возвращает значения PTS в тиках `time_base` (float) для каждого кадра.
        """
        n = len(raw_pts)
        if n == 0:
            return []
        step_ticks = (1.0 / float(nominal_fps)) / float(tb_float)
        known = [i for i, p in enumerate(raw_pts) if p is not None]
        if not known:
            base = float(stream_start_pts) if stream_start_pts is not None else 0.0
            return [base + i * step_ticks for i in range(n)]

        out: List[float] = [float(p) if p is not None else 0.0 for p in raw_pts]

        first = known[0]
        if first > 0:
            p_first = float(raw_pts[first])
            if stream_start_pts is not None and float(stream_start_pts) < p_first:
                a_val = float(stream_start_pts)
                for i in range(first):
                    out[i] = a_val + (p_first - a_val) * i / first
            else:
                floor_val = float(stream_start_pts) if stream_start_pts is not None else 0.0
                back = [p_first - (first - i) * step_ticks for i in range(first)]
                if back[0] < floor_val and floor_val < p_first:
                    back = [floor_val + (p_first - floor_val) * i / first for i in range(first)]
                for i in range(first):
                    out[i] = back[i]

        for a, b in zip(known, known[1:]):
            if b - a > 1:
                pa, pb = float(raw_pts[a]), float(raw_pts[b])
                for i in range(a + 1, b):
                    out[i] = pa + (pb - pa) * (i - a) / (b - a)

        last = known[-1]
        if last < n - 1:
            step = step_ticks
            if len(known) >= 2:
                prev = known[-2]
                local = (float(raw_pts[last]) - float(raw_pts[prev])) / float(last - prev)
                if local > 0:
                    step = local
            for i in range(last + 1, n):
                out[i] = float(raw_pts[last]) + (i - last) * step
        return out

    @staticmethod
    def decode_frames_by_ids(
        video_path: str,
        target_frame_ids: List[str],
        start_offset_sec: float,
        nominal_fps: float,
        max_dim: Optional[int] = None,
        frame_pts_map: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, DecodedFrame]:
        """
        Декодирует из видеопотока ТОЛЬКО запрошенные `target_frame_ids` по их индексу декодирования,
        не удерживая в памяти промежуточные кадры (F05).
        Временная метка каждого кадра берётся из реестра `frame_pts_map` (если передан) либо вычисляется той же
        функцией реконструкции `_reconstruct_missing_pts`, что и при построении реестра (`decode_timeline`):
        для кадра без PTS декодирование продолжается до следующей известной метки (ревью 0909, п. 4.4).
        """
        wanted_set = set(target_frame_ids)
        if not wanted_set:
            return {}

        wanted_idx = sorted(int(fid[1:]) for fid in wanted_set if fid.startswith("F") and fid[1:].isdigit())
        if not wanted_idx:
            return {}
        max_wanted_idx = wanted_idx[-1]
        registry = frame_pts_map or {}

        container = av.open(video_path)
        try:
            stream = container.streams.video[0]
            try:
                stream.thread_type = "AUTO"
            except Exception:
                pass
            tb_fraction: Fraction = stream.time_base
            tb_float = float(tb_fraction)
            stream_start_pts: Optional[int] = int(stream.start_time) if stream.start_time is not None else None

            raw_pts_list: List[Optional[int]] = []
            images: Dict[int, Tuple[np.ndarray, int, int]] = {}
            last_wanted_missing: Optional[int] = None

            for decode_idx, frame in enumerate(container.decode(video=0)):
                fid = f"F{decode_idx:04d}"
                raw_pts_list.append(None if frame.pts is None else int(frame.pts))
                if fid in wanted_set:
                    bgr = frame.to_ndarray(format="bgr24")
                    if max_dim is not None:
                        bgr = _downscale_bgr(bgr, max_dim=max_dim)
                    images[decode_idx] = (bgr, int(frame.width), int(frame.height))
                    if frame.pts is None and fid not in registry:
                        last_wanted_missing = decode_idx
                if decode_idx >= max_wanted_idx:
                    if last_wanted_missing is None:
                        break
                    if decode_idx > last_wanted_missing and frame.pts is not None:
                        break
        finally:
            container.close()

        reconstructed = PyAVVideoDecoder._reconstruct_missing_pts(
            raw_pts_list, tb_float, nominal_fps, stream_start_pts
        )

        result: Dict[str, DecodedFrame] = {}
        for decode_idx, (bgr, orig_w, orig_h) in images.items():
            fid = f"F{decode_idx:04d}"
            meta = registry.get(fid)
            if meta is not None:
                pts_val = int(meta["pts"])
                raw_sec = float(meta["raw_pts_sec"])
                norm_t = float(meta["timestamp_sec"])
            else:
                pts_float = reconstructed[decode_idx]
                pts_val = int(round(pts_float))
                raw_sec = float(pts_float) * tb_float
                norm_t = round(max(0.0, raw_sec - start_offset_sec), 4)
            result[fid] = DecodedFrame(
                frame_id=fid,
                decode_index=decode_idx,
                pts=pts_val,
                raw_pts_sec=round(raw_sec, 4),
                timestamp_sec=norm_t,
                width=orig_w,
                height=orig_h,
                bgr=bgr,
            )
        return result

    @staticmethod
    def decode_timeline(
        video_path: str,
        target_interval_sec: float = 1.0,
        max_coarse_dim: int = 640,
        strict_pts: bool = False,
    ) -> VideoTimelineBundle:
        """
        Выполняет потоковый проход по видеопотоку через PyAV с ограниченным буфером памяти (F05):
        - проверяет целостность контейнера и наличие декодируемого видеопотока;
        - определяет `time_base`, `start_offset_sec`, фактическую длительность по PTS и наличие VFR;
        - сохраняет легковесный реестр PTS/таймкодов (`frame_pts_map`) для всех кадров без конвертации в BGR;
        - материализует массивы изображений (`bgr`) ТОЛЬКО для опорных кадров обзорного прохода (`coarse_frames`)
          с шагом `target_interval_sec` (включая первый и последний кадры).
        При отсутствии части PTS временная шкала восстанавливается единой функцией `_reconstruct_missing_pts`
        (интерполяция между соседними известными метками), которой пользуется и повторное извлечение кадров.
        """
        if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
            raise VideoDecodeError(f"Файл видео не существует или пуст: {video_path}")

        try:
            container = av.open(video_path)
        except Exception as e:
            raise VideoDecodeError(f"Нечитаемый видеоконтейнер ({os.path.basename(video_path)}): {e}") from e

        try:
            if not container.streams.video:
                raise VideoDecodeError(f"В контейнере отсутствует видеопоток: {video_path}")

            stream = container.streams.video[0]
            try:
                stream.thread_type = "AUTO"
            except Exception:
                pass

            if stream.time_base is None:
                raise VideoDecodeError("Видеопоток не содержит корректного time_base.")

            tb_fraction: Fraction = stream.time_base
            tb_float = float(tb_fraction)
            tb_str = f"{tb_fraction.numerator}/{tb_fraction.denominator}"

            avg_rate = float(stream.average_rate) if stream.average_rate else 30.0
            base_rate = float(stream.base_rate) if stream.base_rate else avg_rate
            nominal_fps = avg_rate if avg_rate > 0 else 30.0

            stream_start_pts: Optional[int] = int(stream.start_time) if stream.start_time is not None else None
            stream_start_sec: Optional[float] = (
                float(stream_start_pts * tb_fraction) if stream_start_pts is not None else None
            )

            raw_pts_list: List[Optional[int]] = []
            sampled_raw: Dict[int, np.ndarray] = {}
            next_target_t = 0.0
            last_frame_idx: Optional[int] = None
            last_frame_obj: Optional[av.VideoFrame] = None
            last_sampled_idx = 0
            w0 = 0
            h0 = 0
            missing_pts_count = 0
            prev_prov_sec: Optional[float] = None
            prov_offset: Optional[float] = None
            any_known_seen = False

            decode_idx = 0
            for frame in container.decode(video=0):
                if frame.pts is None:
                    if strict_pts:
                        raise VideoDecodeError(
                            f"Отсутствует временная метка PTS в кадре #{decode_idx} при strict_pts=True"
                        )
                    missing_pts_count += 1
                    raw_pts_list.append(None)
                    # Предварительная оценка времени нужна только для выбора опорных кадров; итоговое время
                    # всех кадров вычисляется после декодирования единой реконструкцией.
                    if prev_prov_sec is not None:
                        prov_sec = prev_prov_sec + (1.0 / nominal_fps)
                    else:
                        prov_sec = (stream_start_sec if stream_start_sec is not None else 0.0) + decode_idx / nominal_fps
                else:
                    pts_int = int(frame.pts)
                    prov_sec = float(pts_int * tb_fraction)
                    if not math.isfinite(prov_sec):
                        raise VideoDecodeError(f"Некорректная временная метка PTS={pts_int} в кадре #{decode_idx}")
                    raw_pts_list.append(pts_int)
                    if not any_known_seen and decode_idx > 0 and stream_start_sec is None:
                        # Первые кадры без PTS и без stream.start_time: переякорение предварительной шкалы
                        prov_offset = prov_sec - decode_idx / nominal_fps
                        next_target_t = (last_sampled_idx / nominal_fps) + target_interval_sec
                    any_known_seen = True

                if decode_idx == 0:
                    w0 = int(frame.width)
                    h0 = int(frame.height)
                    prov_offset = prov_sec
                prev_prov_sec = prov_sec
                norm_prov = max(0.0, prov_sec - (prov_offset or 0.0))

                if decode_idx == 0 or norm_prov >= next_target_t - 1e-3:
                    sampled_raw[decode_idx] = _downscale_bgr(frame.to_ndarray(format="bgr24"), max_dim=max_coarse_dim)
                    next_target_t = norm_prov + target_interval_sec
                    last_sampled_idx = decode_idx
                    last_frame_obj = None
                else:
                    last_frame_idx = decode_idx
                    last_frame_obj = frame

                decode_idx += 1

            if not raw_pts_list:
                raise VideoDecodeError(f"Видеопоток не содержит декодируемых кадров: {video_path}")

            # Обязательно включаем последний декодированный кадр потока в опорные кадры
            final_idx = len(raw_pts_list) - 1
            if final_idx not in sampled_raw and last_frame_obj is not None and last_frame_idx == final_idx:
                sampled_raw[final_idx] = _downscale_bgr(
                    last_frame_obj.to_ndarray(format="bgr24"), max_dim=max_coarse_dim
                )
            last_frame_obj = None

            reconstructed = PyAVVideoDecoder._reconstruct_missing_pts(
                raw_pts_list, tb_float, nominal_fps, stream_start_pts
            )
            raw_secs = [float(p) * tb_float for p in reconstructed]
            for i_r, r_sec in enumerate(raw_secs):
                if not math.isfinite(r_sec):
                    raise VideoDecodeError(f"Некорректная восстановленная временная метка в кадре #{i_r}")

            raw0 = raw_secs[0]
            frame0_reconstructed = raw_pts_list[0] is None
            if stream_start_sec is not None and abs(stream_start_sec - raw0) < 1e-3 and abs(raw0) > 1e-4:
                start_off = raw0
                offset_rule = (
                    f"normalized_by_stream_start_pts (subtracted start_offset_sec={start_off:.4f}s "
                    f"so timeline starts at 0.0s)"
                )
            elif abs(raw0) > 0.05:
                start_off = raw0
                offset_rule = (
                    f"normalized_by_first_decoded_frame_pts (subtracted first_frame_pts_sec={start_off:.4f}s "
                    f"so timeline starts at 0.0s)"
                )
            else:
                start_off = 0.0
                offset_rule = "zero_based_pts (stream starts at 0.0s, no offset subtraction needed)"
            if frame0_reconstructed and start_off:
                offset_rule += " [PTS of frame #0 reconstructed]"

            pts_deltas: List[float] = [
                b - a for a, b in zip(raw_secs, raw_secs[1:]) if (b - a) > 0
            ]
            nonmonotone = [i for i, (a, b) in enumerate(zip(raw_secs, raw_secs[1:])) if b < a]
            if nonmonotone and missing_pts_count > 0:
                # Обратный ход времени при восстановлении недопустим: известные PTS сами по себе немонотонны
                raise VideoDecodeError(
                    f"Невосстановимая временная шкала: немонотонные известные PTS около кадра #{nonmonotone[0]} "
                    f"при {missing_pts_count} отсутствующих метках."
                )

            last_raw_sec = raw_secs[-1]

            # Определение фактической длительности по PTS
            median_frame_dur = float(np.median(pts_deltas)) if pts_deltas else (1.0 / nominal_fps)
            pts_span_duration = (last_raw_sec - start_off) + median_frame_dur
            container_dur = (
                float(container.duration) / float(av.time_base)
                if container.duration is not None
                else 0.0
            )
            stream_dur = (
                float(stream.duration * tb_fraction)
                if stream.duration is not None
                else 0.0
            )
            duration_sec = max(pts_span_duration, stream_dur)
            if container_dur > 0 and abs(container_dur - pts_span_duration) < 1.5:
                duration_sec = max(duration_sec, container_dur)
            duration_sec = round(float(duration_sec), 4)

            # Проверка на VFR (переменную частоту кадров) по разбросу межкадровых интервалов PTS
            is_vfr = False
            if len(pts_deltas) >= 5:
                min_d, max_d = min(pts_deltas), max(pts_deltas)
                if (max_d - min_d) > 0.012 or abs(base_rate - avg_rate) > 1.0:
                    is_vfr = True

            frame_pts_map: Dict[str, Dict[str, Any]] = {}
            coarse_frames: List[DecodedFrame] = []
            retained_frames_map: Dict[str, DecodedFrame] = {}

            for idx, pts_float in enumerate(reconstructed):
                raw_sec = raw_secs[idx]
                pts_val = int(round(pts_float))
                norm_t = round(max(0.0, raw_sec - start_off), 4)
                fid = f"F{idx:04d}"
                frame_pts_map[fid] = {
                    "frame_id": fid,
                    "decode_index": idx,
                    "pts": pts_val,
                    "raw_pts_sec": round(raw_sec, 4),
                    "timestamp_sec": norm_t,
                    "pts_reconstructed": raw_pts_list[idx] is None,
                }
                if idx in sampled_raw:
                    df = DecodedFrame(
                        frame_id=fid,
                        decode_index=idx,
                        pts=pts_val,
                        raw_pts_sec=round(raw_sec, 4),
                        timestamp_sec=norm_t,
                        width=w0,
                        height=h0,
                        bgr=sampled_raw[idx],
                    )
                    coarse_frames.append(df)
                    retained_frames_map[fid] = df

            # Вычисляем фактический максимальный интервал между наблюдениями обзорного прохода декодера/CV
            if len(coarse_frames) >= 2:
                intervals = [
                    coarse_frames[i].timestamp_sec - coarse_frames[i - 1].timestamp_sec
                    for i in range(1, len(coarse_frames))
                ]
                max_interval = round(float(max(intervals)), 4)
            else:
                max_interval = round(float(min(target_interval_sec, duration_sec)), 4)

            pts_fallback_used = missing_pts_count > 0
            if pts_fallback_used:
                offset_rule = (
                    f"pts_reconstructed_by_neighbor_interpolation (missing_pts_frames={missing_pts_count}/{len(raw_pts_list)}, "
                    f"nominal_fps={nominal_fps:.2f}; {offset_rule})"
                )

            limitation_note = (
                f"Первичный обзорный проход выполнен с максимальным шагом {max_interval:.2f} сек по всей длине "
                f"[0.00..{duration_sec:.2f} сек]. Кратковременные микродвижения длительностью менее {max_interval:.2f} сек "
                f"между опорными кадрами могут быть пропущены, если не вызывают изменения состояния сцены или не попадают в окно плотного уточнения."
            )
            if pts_fallback_used:
                limitation_note += (
                    f" ВНИМАНИЕ: в {missing_pts_count} из {len(raw_pts_list)} кадров отсутствовали аппаратные метки PTS; "
                    f"их время восстановлено интерполяцией между соседними известными метками (единый реестр для всех этапов)."
                )

            codec_name = stream.codec_context.name if stream.codec_context else "unknown"

            return VideoTimelineBundle(
                video_path=video_path,
                codec_name=codec_name,
                width=w0,
                height=h0,
                nominal_fps=round(nominal_fps, 3),
                time_base_str=tb_str,
                time_base_float=tb_float,
                start_offset_sec=round(start_off, 4),
                offset_normalization_rule=offset_rule,
                duration_sec=duration_sec,
                is_vfr_detected=is_vfr,
                max_observation_interval_sec=max_interval,
                short_action_limitation_note=limitation_note,
                coarse_frames=coarse_frames,
                all_decoded_frames=retained_frames_map,
                frame_pts_map=frame_pts_map,
                pts_fallback_used=pts_fallback_used,
                missing_pts_frames_count=missing_pts_count,
            )
        finally:
            container.close()

    @staticmethod
    def sample_dense_window(
        bundle: VideoTimelineBundle,
        center_sec: float,
        window_half_sec: float = 4.5,
        max_frames: int = 9,
    ) -> List[DecodedFrame]:
        """
        Извлекает плотную последовательность реально декодированных кадров в окрестности
        `[center_sec - window_half_sec, center_sec + window_half_sec]` с сохранением контекста
        до и после события (Раздел 6.2). Декодирует недостающие кадры окна по требованию (F05).
        """
        t_min = max(0.0, center_sec - window_half_sec)
        t_max = min(bundle.duration_sec, center_sec + window_half_sec)

        pts_entries = sorted(
            (bundle.frame_pts_map or {}).values(),
            key=lambda x: x["decode_index"],
        )
        in_window_meta = [
            m for m in pts_entries if t_min - 1e-3 <= float(m["timestamp_sec"]) <= t_max + 1e-3
        ]
        if not in_window_meta:
            in_window_meta = pts_entries

        if len(in_window_meta) <= max_frames:
            chosen_meta = in_window_meta
        else:
            indices = np.linspace(0, len(in_window_meta) - 1, num=max_frames, dtype=int)
            chosen_meta = []
            seen_ids = set()
            for i in indices:
                m = in_window_meta[int(i)]
                if m["frame_id"] not in seen_ids:
                    seen_ids.add(m["frame_id"])
                    chosen_meta.append(m)

        missing_fids = [
            m["frame_id"]
            for m in chosen_meta
            if m["frame_id"] not in bundle.all_decoded_frames
        ]
        on_demand_decoded: Dict[str, DecodedFrame] = {}
        if missing_fids and os.path.exists(bundle.video_path):
            on_demand_decoded = PyAVVideoDecoder.decode_frames_by_ids(
                video_path=bundle.video_path,
                target_frame_ids=missing_fids,
                start_offset_sec=bundle.start_offset_sec,
                nominal_fps=bundle.nominal_fps,
                max_dim=640,
                frame_pts_map=bundle.frame_pts_map,
            )

        selected: List[DecodedFrame] = []
        for m in chosen_meta:
            fid = m["frame_id"]
            if fid in bundle.all_decoded_frames:
                selected.append(bundle.all_decoded_frames[fid])
            elif fid in on_demand_decoded:
                selected.append(on_demand_decoded[fid])
        return selected

    @staticmethod
    def build_annotated_storyboard(
        frames: List[DecodedFrame],
        tile_size: Tuple[int, int] = (320, 420),
        cols: int = 3,
    ) -> Image.Image:
        """
        Строит раскадровку (contact sheet) из переданных декодированных кадров.
        Заголовок с `frame_id` и точным PTS-таймкодом `t=XX.XXs` размещается в отдельной
        верхней плашке над каждым кадром, чтобы не перекрывать полезное изображение сцены.
        """
        if not frames:
            raise ValueError("Список кадров для построения раскадровки пуст.")

        tw, th = tile_size
        header_h = 28
        cell_h = th + header_h
        n = len(frames)
        cols = max(1, min(cols, n))
        rows = int(math.ceil(n / cols))

        canvas = Image.new("RGB", (cols * tw, rows * cell_h), (18, 22, 30))
        draw = ImageDraw.Draw(canvas)

        for idx, df in enumerate(frames):
            r = idx // cols
            c = idx % cols
            x0 = c * tw
            y0 = r * cell_h

            # Плашка заголовка кадра (не перекрывает сам кадр)
            draw.rectangle([x0, y0, x0 + tw - 1, y0 + header_h - 1], fill=(28, 35, 48))
            label = f"ID:{df.frame_id} | t={df.timestamp_sec:.2f}s (PTS:{df.pts})"
            draw.text((x0 + 8, y0 + 6), label, fill=(255, 230, 120))

            pil_im = df.to_pil().resize((tw, th), Image.Resampling.BILINEAR)
            canvas.paste(pil_im, (x0, y0 + header_h))
            draw.rectangle(
                [x0, y0, x0 + tw - 1, y0 + cell_h - 1],
                outline=(70, 82, 105),
                width=1,
            )

        return canvas

    @staticmethod
    def encode_verified_evidence_frames(
        bundle: VideoTimelineBundle,
        frame_ids: List[str],
        jpeg_quality: int = 88,
    ) -> List[Dict[str, Any]]:
        """
        Преобразует 1–3 выбранных `frame_id` в проверенные доказательные кадры JPEG base64.
        Проверяет:
        1. Каждый `frame_id` действительно принадлежит декодированному пулу кадров видео (`frame_pts_map` / `all_decoded_frames`, Раздел 8.2, AT-13).
        2. Временная метка `timestamp_sec` конечна и строго лежит в пределах `[0, duration_sec]` без молчаливого обрезания.
        3. При наличии исходного видеофайла декодирует выбранные 1–3 кадра в оригинальном разрешении по требованию (F05).
        4. Сгенерированный base64 успешно декодируется обратно в валидное JPEG-изображение (Раздел 9, AT-17).
        """
        if not frame_ids:
            raise ValueError("Список доказательных frame_id не может быть пустым.")

        known_ids = set((bundle.frame_pts_map or {}).keys()) | set(bundle.all_decoded_frames.keys())
        dedup_ids: List[str] = []
        for fid in frame_ids:
            if fid not in known_ids:
                raise ValueError(
                    f"Недопустимый frame_id '{fid}': кадр отсутствует среди реально декодированных кадров видео."
                )
            if fid not in dedup_ids:
                dedup_ids.append(fid)

        selected_ids = dedup_ids[:3]
        fullres_decoded: Dict[str, DecodedFrame] = {}
        if os.path.exists(bundle.video_path):
            try:
                fullres_decoded = PyAVVideoDecoder.decode_frames_by_ids(
                    video_path=bundle.video_path,
                    target_frame_ids=selected_ids,
                    start_offset_sec=bundle.start_offset_sec,
                    nominal_fps=bundle.nominal_fps,
                    max_dim=None,
                    frame_pts_map=bundle.frame_pts_map,
                )
            except Exception:
                fullres_decoded = {}

        evidence_out: List[Dict[str, Any]] = []

        for fid in selected_ids:
            df = fullres_decoded.get(fid) or bundle.all_decoded_frames.get(fid)
            if df is None:
                raise ValueError(f"Не удалось извлечь изображение доказательного кадра '{fid}'.")

            ts = float(df.timestamp_sec)
            if not math.isfinite(ts) or ts < 0.0 or ts > bundle.duration_sec:
                raise ValueError(
                    f"Временная метка доказательного кадра {fid} ({ts}s) выходит за пределы видео [0, {bundle.duration_sec}s]."
                )

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
            ok, buf = cv2.imencode(".jpg", df.bgr, encode_param)
            if not ok:
                raise ValueError(f"Ошибка кодирования JPEG для доказательного кадра {fid}.")

            raw_bytes = buf.tobytes()
            b64_str = base64.b64encode(raw_bytes).decode("ascii")

            # Техническая верификация обратимости base64 -> JPEG (Раздел 9)
            decoded_bytes = base64.b64decode(b64_str, validate=True)
            with Image.open(io.BytesIO(decoded_bytes)) as verify_img:
                verify_img.verify()

            evidence_out.append(
                {
                    "frame_id": fid,
                    "pts": df.pts,
                    "timestamp_sec": round(ts, 2),
                    "image_base64": b64_str,
                }
            )

        return evidence_out
