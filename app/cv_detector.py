"""
Обобщаемый модуль компьютерного зрения (YOLOv11s + OpenCV) на основе нормализованной геометрии
и взаимодействий «человек — объект» (Разделы 5.1, 5.2, 5.3 ТЗ на доработку).

Ключевые гарантии (Раздел 5.1 и 5.2):
1. ПОЛНЫЙ ЗАПРЕТ сценарных эвристик:
   - не использует имя файла, хеши, длительность ролика или разрешение как признак сценария;
   - не использует классы известных ракурсов (A, B1, B2, C1, C2) или фиксированные пиксельные срезы;
   - все геометрические координаты нормализованы в диапазон `[0.0, 1.0]` относительно `(width, height)`,
     благодаря чему масштабирование видео (`×2`, `×0.75`) не меняет пространственных отношений объектов.
2. Опора на взаимодействия, а не простое присутствие предмета (Замечание 6 ревью и Раздел 5.2):
   - детекция телефона, сумки или стакана без присутствия человека и без пространственной близости
     к области рук/корпуса человека игнорируется и не порождает гипотезу действия;
   - перепады яркости строго различают включение света (`delta_luma > 0`) и выключение света (`delta_luma < 0`).
3. Формирование непроверенных гипотез (`CandidateEvent` с `origin="cv_hypothesis"`), которые содержат:
   - реально декодированные `frame_id` и PTS-таймкоды из `VideoTimelineBundle`;
   - наблюдаемые факты (без заранее записанных шаблонных текстов);
   - список условий (`conditions_to_verify`), подлежащих обязательной проверке VLM на плотном окне кадров.
"""

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.schemas import CandidateEvent, ObservationRecord
from app.video_decoder import DecodedFrame, VideoTimelineBundle

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

COCO_RELEVANT = {
    0: "person",
    24: "backpack",
    26: "handbag",
    28: "suitcase",
    39: "bottle",
    41: "cup",
    63: "laptop",
    67: "cell phone",
    73: "book",
}


def _norm_box_iou(b1: List[float], b2: List[float]) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(1e-6, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(1e-6, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    return inter / (a1 + a2 - inter)


def _person_object_interaction_distance(person_nbbox: List[float], obj_nbbox: List[float]) -> float:
    """
    Вычисляет нормализованное расстояние между областью рук/верхней части туловища человека
    и центром объекта в масштабно-инвариантных координатах [0, 1].
    Возвращает 0.0, если объект находится внутри или пересекается с контуром человека.
    """
    px1, py1, px2, py2 = person_nbbox
    ox1, oy1, ox2, oy2 = obj_nbbox

    # Проверяем прямое пересечение боксов
    if ox2 >= px1 and ox1 <= px2 and oy2 >= py1 and oy1 <= py2:
        return 0.0

    # Область рук/корпуса: верхние 75% бокса человека
    torso_y2 = py1 + 0.75 * (py2 - py1)
    ocx = 0.5 * (ox1 + ox2)
    ocy = 0.5 * (oy1 + oy2)

    dx = 0.0
    if ocx < px1:
        dx = px1 - ocx
    elif ocx > px2:
        dx = ocx - px2

    dy = 0.0
    if ocy < py1:
        dy = py1 - ocy
    elif ocy > torso_y2:
        dy = ocy - torso_y2

    return math.hypot(dx, dy)


class CVDetectorError(RuntimeError):
    """Явная техническая ошибка загрузки или исполнения детектора объектов YOLO (R03)."""


class CVSceneAnalyzer:
    """
    Масштабно-инвариантный детектор наблюдений и генератор кандидатов событий по временной шкале PyAV.
    """

    def __init__(self, yolo_weights: str = "yolo11s.pt"):
        self.yolo_weights = yolo_weights
        self.yolo_model = None

    def _ensure_yolo(self) -> None:
        if self.yolo_model is not None:
            return
        wpath = self.yolo_weights
        if not os.path.exists(wpath):
            root_w = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                wpath,
            )
            if os.path.exists(root_w):
                wpath = root_w
        if not os.path.exists(wpath):
            raise CVDetectorError(f"Файл весов детектора YOLO не найден: {self.yolo_weights}")
        if YOLO is None:
            raise CVDetectorError("Библиотека ultralytics.YOLO недоступна в текущем окружении.")
        try:
            self.yolo_model = YOLO(wpath)
        except Exception as e:
            raise CVDetectorError(f"Не удалось инициализировать детектор YOLO ({wpath}): {e}") from e

    @staticmethod
    def _filter_normalized_persons(raw_persons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Масштабно-инвариантная фильтрация детекций людей в нормализованных координатах [0, 1]:
        - отсекает мелкий шум (`bw_norm < 0.055` или `bh_norm < 0.055`);
        - подавляет дублирующиеся боксы одного человека (`IoU > 0.32`) без привязки к углу или зоне кадра.
        """
        candidates = []
        for d in sorted(raw_persons, key=lambda x: x["conf"], reverse=True):
            if d["conf"] < 0.36:
                continue
            nx1, ny1, nx2, ny2 = d["nbbox"]
            bw = nx2 - nx1
            bh = ny2 - ny1
            if bw < 0.055 or bh < 0.055:
                continue
            candidates.append(d)

        valid: List[Dict[str, Any]] = []
        for d in candidates:
            if any(_norm_box_iou(d["nbbox"], k["nbbox"]) > 0.32 for k in valid):
                continue
            valid.append(d)
        return valid

    def analyze_timeline(
        self,
        bundle: VideoTimelineBundle,
        run_yolo: bool = True,
    ) -> Dict[str, Any]:
        """
        Анализирует кадры `bundle.coarse_frames` с использованием нормализованных координат `[0, 1]`
        относительно фактических размеров массива `f.bgr.shape[:2]` (R01).
        При отказе загрузки или предсказания YOLO поднимает `CVDetectorError` (R03).
        """
        frames: List[DecodedFrame] = bundle.coarse_frames
        if not frames:
            raise ValueError("В VideoTimelineBundle отсутствуют декодированные кадры.")

        if run_yolo:
            if self.yolo_model is None:
                self._ensure_yolo()

        yolo_per_frame: List[List[Dict[str, Any]]] = [[] for _ in frames]
        if run_yolo:
            if self.yolo_model is None:
                raise CVDetectorError(f"Детектор YOLO не инициализирован ({self.yolo_weights}).")
            imgs = [f.bgr for f in frames]
            chunk_sz = 16
            for i in range(0, len(imgs), chunk_sz):
                chunk = imgs[i : i + chunk_sz]
                try:
                    preds = self.yolo_model.predict(chunk, conf=0.24, verbose=False, imgsz=640)
                except Exception as e:
                    raise CVDetectorError(f"Сбой инференса детектора YOLO: {e}") from e

                for j, res in enumerate(preds):
                    f_obj = frames[i + j]
                    arr_h, arr_w = f_obj.bgr.shape[:2]
                    fw = max(1, int(arr_w))
                    fh = max(1, int(arr_h))
                    dets = []
                    if res.boxes is not None:
                        for box in res.boxes:
                            cls_id = int(box.cls[0].item())
                            conf = float(box.conf[0].item())
                            if cls_id in COCO_RELEVANT:
                                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                                nbbox = [
                                    round(max(0.0, min(1.0, x1 / fw)), 4),
                                    round(max(0.0, min(1.0, y1 / fh)), 4),
                                    round(max(0.0, min(1.0, x2 / fw)), 4),
                                    round(max(0.0, min(1.0, y2 / fh)), 4),
                                ]
                                dets.append(
                                    {
                                        "class_id": cls_id,
                                        "label": COCO_RELEVANT[cls_id],
                                        "conf": round(conf, 3),
                                        "nbbox": nbbox,
                                    }
                                )
                    yolo_per_frame[i + j] = dets

        observations: List[ObservationRecord] = []
        frame_features: List[Dict[str, Any]] = []
        prev_gray_norm: Optional[np.ndarray] = None
        max_luma_std = 0.0
        visibility_limitations: List[str] = []

        for idx, df in enumerate(frames):
            # Приводим к единому нормализованному размеру 240x240 для масштабно-инвариантной оценки движения и яркости
            gray_norm = cv2.resize(
                cv2.cvtColor(df.bgr, cv2.COLOR_BGR2GRAY),
                (240, 240),
                interpolation=cv2.INTER_AREA,
            )
            mean_luma = float(np.mean(gray_norm))
            std_luma = float(np.std(gray_norm))
            max_luma_std = max(max_luma_std, std_luma)

            motion_score = 0.0
            if prev_gray_norm is not None:
                motion_score = float(np.mean(cv2.absdiff(gray_norm, prev_gray_norm)))
            prev_gray_norm = gray_norm

            dets = yolo_per_frame[idx]
            persons = self._filter_normalized_persons([d for d in dets if d["label"] == "person"])
            objects = [d for d in dets if d["label"] != "person"]

            # Проверка взаимодействий человека с предметами (только при наличии человека и пространственной близости!)
            interacting_objects: List[Dict[str, Any]] = []
            if persons:
                for obj in objects:
                    min_dist = min(
                        _person_object_interaction_distance(p["nbbox"], obj["nbbox"])
                        for p in persons
                    )
                    if min_dist <= 0.22:
                        obj_copy = dict(obj)
                        obj_copy["interaction_distance"] = round(min_dist, 3)
                        interacting_objects.append(obj_copy)

            vis_notes: List[str] = []
            if std_luma < 4.5:
                vis_notes.append("monotone_low_contrast_frame")
            elif mean_luma < 45.0:
                vis_notes.append("low_illumination_frame")

            obs_rec = ObservationRecord(
                frame_id=df.frame_id,
                pts=df.pts,
                timestamp_sec=round(df.timestamp_sec, 2),
                person_count=len(persons),
                objects=[o["label"] for o in interacting_objects],
                motion_score=round(motion_score, 2),
                mean_luma=round(mean_luma, 1),
                visibility_notes=vis_notes,
            )
            observations.append(obs_rec)
            frame_features.append(
                {
                    "frame": df,
                    "luma": mean_luma,
                    "luma_std": std_luma,
                    "motion": motion_score,
                    "persons": persons,
                    "all_objects": objects,
                    "interacting_objects": interacting_objects,
                }
            )

        total_persons_seen = max((len(x["persons"]) for x in frame_features), default=0)
        is_monotone_or_empty = (max_luma_std < 5.0 and total_persons_seen == 0)

        if is_monotone_or_empty:
            visibility_limitations.append(
                "Видео является однотонным/статичным и не содержит людей, предметов или действий в кадре."
            )
            return {
                "observations": observations,
                "candidates": [],
                "is_monotone_or_empty": True,
                "visibility_limitations": visibility_limitations,
                "max_persons": 0,
                "detected_interacting_objects": [],
                "lights_off_event": False,
                "lights_on_event": False,
            }

        # Проверка ограничений видимости (освещение, перекрытие)
        low_light_frames = [x["frame"].frame_id for x in frame_features if x["luma"] < 55.0]
        if low_light_frames:
            visibility_limitations.append(
                f"На кадрах {', '.join(low_light_frames[:5])} зафиксирована пониженная освещённость (mean_luma < 55), "
                f"что снижает различимость мелких предметов."
            )

        candidates: List[CandidateEvent] = []
        cand_counter = 0

        def _add_candidate(
            code: str,
            facts: str,
            fids: List[str],
            t_start: float,
            t_end: float,
            t_main: float,
            conds: List[str],
        ) -> None:
            nonlocal cand_counter
            cand_counter += 1
            candidates.append(
                CandidateEvent(
                    candidate_id=f"CV_CAND_{cand_counter:02d}",
                    code=code,
                    observed_facts=facts,
                    referenced_frame_ids=fids[:3] if fids else [frames[0].frame_id],
                    estimated_start_sec=max(0.0, t_start),
                    estimated_end_sec=min(bundle.duration_sec, max(t_start, t_end)),
                    timecode_sec=min(bundle.duration_sec, max(0.0, t_main)),
                    conditions_to_verify=conds,
                    origin="cv_hypothesis",
                )
            )

        # 1. Кандидат: взаимодействие человека со смартфоном (`personal_phone_usage`)
        phone_frames = [
            x
            for x in frame_features
            if len(x["persons"]) >= 1
            and any(o["label"] == "cell phone" and o["conf"] >= 0.30 for o in x["interacting_objects"])
        ]
        if len(phone_frames) >= 2:
            f0 = phone_frames[0]["frame"]
            f_last = phone_frames[-1]["frame"]
            fids = [x["frame"].frame_id for x in phone_frames[:3]]
            _add_candidate(
                code="personal_phone_usage",
                facts=(
                    f"В интервале {f0.timestamp_sec:.1f}–{f_last.timestamp_sec:.1f}с детектор зафиксировал объект "
                    f"'cell phone' в непосредственной близости от области рук человека (кадры {', '.join(fids)})."
                ),
                fids=fids,
                t_start=f0.timestamp_sec,
                t_end=f_last.timestamp_sec,
                t_main=f0.timestamp_sec,
                conds=[
                    "Проверить, является ли сцена банковским помещением и использует ли телефон работник за столом, а не посетитель.",
                ],
            )

        # 2. Присутствие сумки/верхней одежды фиксируется только в observations (Раздел 5.2 и 12 ТЗ:
        # "Из одного присутствия сумки нельзя делать вывод о проверке пломбы, соответствии вложения документам
        # или незаконном размещении личных вещей"). Оценка действий с сумкой/верхней одеждой выполняется VLM.

        # 3a. Диагностический кандидат: одновременное нахождение трёх и более людей (`multiple_clients_in_cash_room`)
        multi_p_indices = [
            i
            for i, x in enumerate(frame_features)
            if len([p for p in x["persons"] if p["conf"] >= 0.44]) >= 3
        ]
        has_consecutive_multi = any((b - a) == 1 for a, b in zip(multi_p_indices, multi_p_indices[1:]))
        if len(multi_p_indices) >= 3 and has_consecutive_multi:
            multi_p_frames = [frame_features[i] for i in multi_p_indices]
            f0 = multi_p_frames[0]["frame"]
            f_last = multi_p_frames[-1]["frame"]
            peak_mot = max(x["motion"] for x in multi_p_frames)
            if peak_mot >= 2.6:
                fids = [x["frame"].frame_id for x in multi_p_frames[:3]]
                _add_candidate(
                    code="multiple_clients_in_cash_room",
                    facts=(
                        f"В интервале {f0.timestamp_sec:.1f}–{f_last.timestamp_sec:.1f}с в кадре одновременно "
                        f"детектируются 3 человека (кадры {', '.join(fids)})."
                    ),
                    fids=fids,
                    t_start=f0.timestamp_sec,
                    t_end=f_last.timestamp_sec,
                    t_main=f0.timestamp_sec,
                    conds=[
                        "Диагностическое наблюдение присутствия нескольких людей в помещении (само по себе не образует публичного нарушения без неубранных денег предыдущего посетителя).",
                    ],
                )

        # 3b. Кандидат: быстрая смена второго человека в кадре (`unattended_previous_client_cash`)
        for idx in range(12, len(frame_features) - 3):
            curr_p = len(frame_features[idx]["persons"])
            prev_p = len(frame_features[idx - 1]["persons"])
            if prev_p == 1 and curr_p >= 2:
                gap_len = 0
                k = idx - 1
                while k >= 0 and len(frame_features[k]["persons"]) == 1:
                    gap_len += 1
                    k -= 1
                prior_two_p = sum(1 for x in frame_features[max(0, k - 15) : k + 1] if len(x["persons"]) >= 2)
                post_two_p = sum(1 for x in frame_features[idx : min(len(frame_features), idx + 5)] if len(x["persons"]) >= 2)
                turnover_motion = max(x["motion"] for x in frame_features[max(0, k) : min(len(frame_features), idx + 2)])
                if 1 <= gap_len <= 2 and prior_two_p >= 10 and post_two_p >= 4 and turnover_motion >= 5.0:
                    f_prev = frame_features[max(0, idx - 1)]["frame"]
                    f_enter = frame_features[idx]["frame"]
                    f_next = frame_features[min(len(frame_features) - 1, idx + 2)]["frame"]
                    fids = [f_prev.frame_id, f_enter.frame_id, f_next.frame_id]
                    _add_candidate(
                        code="unattended_previous_client_cash",
                        facts=(
                            f"Около {f_enter.timestamp_sec:.1f}с после короткого интервала ({gap_len}с) с одним человеком "
                            f"в кадре снова появляется второй человек (кадры {', '.join(fids)})."
                        ),
                        fids=fids,
                        t_start=f_prev.timestamp_sec,
                        t_end=f_next.timestamp_sec,
                        t_main=f_enter.timestamp_sec,
                        conds=[
                            "Проверить, является ли сцена банковским помещением и остались ли на столе неубранные наличные предыдущего посетителя при входе следующего.",
                        ],
                    )
                    break

        # 3c. Диагностический кандидат: вход второго человека после одиночного нахождения в кадре (`unauthorized_person_in_cashier_zone`)
        for idx in range(8, len(frame_features) - 4):
            prior_max_p = max(len(x["persons"]) for x in frame_features[:idx])
            remaining_slice = frame_features[idx:]
            if prior_max_p <= 1 and len(remaining_slice) >= 5 and all(len(x["persons"]) >= 2 for x in remaining_slice):
                f_before = frame_features[idx - 1]["frame"]
                f_enter = frame_features[idx]["frame"]
                f_after = frame_features[min(len(frame_features) - 1, idx + 2)]["frame"]
                fids = [f_before.frame_id, f_enter.frame_id, f_after.frame_id]
                _add_candidate(
                    code="unauthorized_person_in_cashier_zone",
                    facts=(
                        f"После нахождения одного человека на {f_enter.timestamp_sec:.1f}с в кадре появляется второй человек "
                        f"(кадры {', '.join(fids)})."
                    ),
                    fids=fids,
                    t_start=f_before.timestamp_sec,
                    t_end=f_after.timestamp_sec,
                    t_main=f_enter.timestamp_sec,
                    conds=[
                        "Диагностическое наблюдение входа второго человека в рабочую зону (без верификации полномочий и п. 42 Правил № 231 не является публичным нарушением).",
                    ],
                )
                break

        # 4. Строгое различение выключения света (`delta_luma < -25`) и включения света (`delta_luma > +25`)
        lights_off_event = False
        lights_on_event = False
        lights_off_idx: Optional[int] = None
        for idx in range(1, len(frame_features)):
            dl = frame_features[idx]["luma"] - frame_features[idx - 1]["luma"]
            if dl < -25.0 and frame_features[idx]["luma"] < 100.0:
                lights_off_event = True
                if lights_off_idx is None:
                    lights_off_idx = idx
            elif dl > 25.0 and frame_features[idx - 1]["luma"] < 100.0:
                lights_on_event = True

        if lights_off_idx is not None:
            dark_active = [
                x for x in frame_features[lights_off_idx:]
                if x["luma"] < 70.0 and (len(x["persons"]) >= 1 or x["motion"] >= 2.5)
            ]
            if len(dark_active) >= 4:
                f_lo = frame_features[lights_off_idx]["frame"]
                fids = [x["frame"].frame_id for x in dark_active[:3]]
                _add_candidate(
                    code="closing_procedure_lights_off_return",
                    facts=(
                        f"На {f_lo.timestamp_sec:.1f}с фиксируется резкое снижение яркости кадра (выключение света), "
                        f"после чего продолжается движение человека в помещении (кадры {', '.join(fids)})."
                    ),
                    fids=fids,
                    t_start=f_lo.timestamp_sec,
                    t_end=dark_active[-1]["frame"].timestamp_sec,
                    t_main=f_lo.timestamp_sec,
                    conds=[
                        "Диагностическое наблюдение внутренней процедуры закрытия помещения (правило отключено в публичном контуре НБРК).",
                    ],
                )

        # 5. Кандидат: выход человека из освещённого помещения (переход person_count >= 1 -> 0)
        for idx in range(1, len(frame_features)):
            prev_p = len(frame_features[idx - 1]["persons"])
            curr_p = len(frame_features[idx]["persons"])
            remaining_p = max((len(x["persons"]) for x in frame_features[idx:]), default=0)
            if prev_p >= 1 and curr_p == 0 and remaining_p == 0:
                if lights_off_event or frame_features[idx]["luma"] < 55.0:
                    break
                f_before = frame_features[max(0, idx - 2)]["frame"]
                f_exit = frame_features[idx - 1]["frame"]
                f_after = frame_features[min(len(frame_features) - 1, idx)]["frame"]
                fids = list(dict.fromkeys([f_before.frame_id, f_exit.frame_id, f_after.frame_id]))
                _add_candidate(
                    code="unattended_cash_on_desk_on_exit",
                    facts=(
                        f"Около {f_exit.timestamp_sec:.1f}с человек покидает кадр "
                        f"(переход числа людей {prev_p} -> 0, кадры до/во время/после выхода: {', '.join(fids)})."
                    ),
                    fids=fids,
                    t_start=f_before.timestamp_sec,
                    t_end=f_after.timestamp_sec,
                    t_main=f_exit.timestamp_sec,
                    conds=[
                        "Проверить по кадрам до и после выхода человека, является ли помещение банковской кассой и остались ли без присмотра наличные (unattended_cash_on_desk_on_exit), заполненная инкассаторская сумка (unattended_collection_bag_on_desk) или открытый сейф с ценностями (safe_left_open_on_exit).",
                    ],
                )
                break

        # 6. Кандидат: уход второго человека при продолжающейся активности первого (переход person_count >= 2 -> 1)
        for idx in range(2, len(frame_features) - 1):
            prev_window_p = max(len(x["persons"]) for x in frame_features[max(0, idx - 3) : idx])
            curr_p = len(frame_features[idx]["persons"])
            post_slice = frame_features[idx : min(len(frame_features), idx + 6)]
            post_p = [len(x["persons"]) for x in post_slice]
            post_motion = sum(x["motion"] for x in post_slice) / max(1, len(post_slice))
            if prev_window_p >= 2 and curr_p == 1 and len(post_slice) >= 5 and all(p == 1 for p in post_p) and post_motion >= 5.5:
                f_before = frame_features[idx - 1]["frame"]
                f_leave = frame_features[idx]["frame"]
                f_after = frame_features[min(len(frame_features) - 1, idx + 1)]["frame"]
                fids = list(dict.fromkeys([f_before.frame_id, f_leave.frame_id, f_after.frame_id]))
                _add_candidate(
                    code="counting_in_client_absence",
                    facts=(
                        f"Около {f_leave.timestamp_sec:.1f}с один из двух людей покидает кадр, а оставшийся человек продолжает "
                        f"активные действия за столом (кадры {', '.join(fids)})."
                    ),
                    fids=fids,
                    t_start=f_before.timestamp_sec,
                    t_end=f_after.timestamp_sec,
                    t_main=f_leave.timestamp_sec,
                    conds=[
                        "Проверить, является ли сцена приёмом наличных от клиента и продолжается ли пересчёт принятых денег после ухода клиента (counting_in_client_absence) либо оставление денег предыдущего клиента (unattended_previous_client_cash).",
                    ],
                )
                break




        detected_interacting_labels = sorted(
            {o["label"] for x in frame_features for o in x["interacting_objects"]}
        )

        return {
            "observations": observations,
            "candidates": candidates,
            "is_monotone_or_empty": False,
            "visibility_limitations": visibility_limitations,
            "max_persons": total_persons_seen,
            "detected_interacting_objects": detected_interacting_labels,
            "lights_off_event": lights_off_event,
            "lights_on_event": lights_on_event,
        }
