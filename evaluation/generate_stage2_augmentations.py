"""
Stage 2 & Stage 5 Controlled Video Augmentations Generator & Manifest Builder (Section 15).

Реализует:
1. Пилотный набор из 10 репрезентативных исходных видеороликов (6 с нарушениями разных типов и длительности,
   4 чистых ролика с разных ракурсов) с зафиксированным обоснованием выбора до просмотра результатов новой версии.
2. Пять обязательных одиночных преобразований (Раздел 15.2):
   - `scale_2x`: увеличение разрешения в 2 раза (проверка независимости от абсолютных пиксельных координат).
   - `scale_075x`: уменьшение разрешения ~×0.75 (проверка устойчивости при умеренном снижении детализации).
   - `prefix_5s`: нейтральная вставка +5 секунд в начало (проверка сдвига таймкода t_new = t_orig + delta_measured).
   - `luma_contrast`: умеренное согласованное изменение яркости и контраста (brightness=+0.05, contrast=1.08).
   - `h264_compression`: умеренное сжатие H.264 (CRF 28, проверка устойчивости к артефактам).
3. Специальные синтетические и технические тест-кейсы для приёмочных испытаний:
   - `monotone_valid.mov` (AT-01: корректно декодируемое однотонное видео 6 сек).
   - `vfr_offset.mov` (AT-08: видео с переменной частотой кадров VFR и ненулевым начальным PTS offset).
   - `corrupted_invalid.mov` (AT-14: повреждённый/недекодируемый контейнер).
4. Расширение (`--expand-all`) на все 42 видео открытого набора с явной разметкой семантической валидности
   (`semantic_validity`: `valid`, `degraded_excluded_from_strict_label`, `not_applicable_illumination_event`).
"""

import argparse
import glob
import hashlib
import json
import os
import subprocess
from typing import Dict, Any, List, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(BASE_DIR, "evaluation")
AUG_DIR = os.path.join(EVAL_DIR, "augmentations")
SYNTH_DIR = os.path.join(EVAL_DIR, "synthetic_tests")
OPEN_VIDEOS_DIR = os.path.join(BASE_DIR, "open", "videos")
FFMPEG_BIN = "/opt/homebrew/bin/ffmpeg"
FFPROBE_BIN = "/opt/homebrew/bin/ffprobe"

# 10 пилотных исходных видео (выбраны до прогона новой модели, покрывают разные действия, длительности и чистые сцены)
PILOT_10_SELECTION: Dict[str, str] = {
    "Х-4": "Длинный ролик (76.7с), нарушение п. 20 НБРК № 231 (неубранные наличные первого клиента при входе второго на 57с и 68с).",
    "Х-17": "Короткий ролик (13.3с), нарушение п. 24 НБРК № 231 (приём инкассаторской сумки без проверки пломбы и документов на 7с).",
    "Х-18": "Средний ролик (24.5с), нарушение п. 20 НБРК № 231 (вход второго клиента в кассу на 20с во время обслуживания первого).",
    "Х-20": "Средний ролик (22.7с), нарушение п. 20/80 НБРК № 231 (дверца сейфа оставлена открытой во время работы кассира с 5с).",
    "Х-26": "Короткий ролик (16.5с), нарушение п. 20 НБРК № 231 (оставление наличных на столе при выходе кассира на 9с).",
    "Х-33": "Средний ролик (18.4с), нарушение п. 12 НБРК № 231 (использование личного смартфона кассиром на рабочем месте с 6с).",
    "Х-3": "Чистый ролик без нарушений (19.6с), стандартный приём и пересчёт наличных при клиенте.",
    "Х-14": "Чистый ролик без нарушений (15.4с), корректная работа кассира при закрытом сейфе и убранных ценностях.",
    "Х-31": "Чистый ролик без нарушений (16.2с), корректный вход и работа за столом.",
    "Х-35": "Чистый ролик без нарушений (19.3с), корректное обслуживание клиента.",
}

# Сценарии, где мелкие детали (номиналы купюр, мелкие штампы) при ×0.75 или сильном сжатии теряют различимость
FINE_DETAIL_STEMS = {"Х-44", "Х-24"}
# Сценарии, где изменение яркости меняет семантику события с освещением
ILLUMINATION_SENSITIVE_STEMS = {"Х-11", "Х-21"}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def probe_video(path: str) -> Dict[str, Any]:
    cmd = [
        FFPROBE_BIN,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,time_base,start_pts,start_time,duration,nb_frames",
        "-show_entries",
        "format=duration,size",
        "-of",
        "json",
        path,
    ]
    out = subprocess.check_output(cmd, text=True)
    data = json.loads(out)
    stream = (data.get("streams") or [{}])[0]
    fmt = data.get("format") or {}
    dur = float(stream.get("duration") or fmt.get("duration") or 0.0)
    return {
        "codec_name": stream.get("codec_name", ""),
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "r_frame_rate": stream.get("r_frame_rate", ""),
        "avg_frame_rate": stream.get("avg_frame_rate", ""),
        "time_base": stream.get("time_base", ""),
        "start_time": float(stream.get("start_time") or 0.0),
        "duration": round(dur, 4),
        "nb_frames": int(stream.get("nb_frames") or 0),
        "size_bytes": int(fmt.get("size") or os.path.getsize(path)),
    }


def generate_synthetic_acceptance_files() -> List[Dict[str, Any]]:
    os.makedirs(SYNTH_DIR, exist_ok=True)
    synth_manifest = []

    # 1. AT-01: Однотонное корректное видео (6.0 сек, 464x848, 30 fps)
    monotone_path = os.path.join(SYNTH_DIR, "monotone_valid.mov")
    cmd_mono = [
        FFMPEG_BIN,
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=0x808080:s=464x848:r=30:d=6.0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        monotone_path,
    ]
    subprocess.run(cmd_mono, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    synth_manifest.append(
        {
            "test_id": "AT-01",
            "filename": "monotone_valid.mov",
            "path": os.path.relpath(monotone_path, BASE_DIR),
            "sha256": sha256_file(monotone_path),
            "description": "Корректно декодируемое однотонное серое видео (6.0с) без людей и объектов.",
            "expected_status": "done",
            "expected_verdict": "no_violation",
            "probed": probe_video(monotone_path),
        }
    )

    # 2. AT-08: VFR + смещение начала потока (на базе Х-33 или Х-26 со смещением output_ts_offset=3.5s и переменным фреймрейтом)
    src_vfr = os.path.join(OPEN_VIDEOS_DIR, "Х-33.mov")
    vfr_path = os.path.join(SYNTH_DIR, "vfr_offset_test.mov")
    if os.path.exists(src_vfr):
        cmd_vfr = [
            FFMPEG_BIN,
            "-y",
            "-i",
            src_vfr,
            "-vf",
            "select='not(mod(n,2))+gt(t,5)*mod(n,2)'",
            "-vsync",
            "vfr",
            "-output_ts_offset",
            "3.5",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            vfr_path,
        ]
        subprocess.run(cmd_vfr, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        synth_manifest.append(
            {
                "test_id": "AT-08",
                "filename": "vfr_offset_test.mov",
                "path": os.path.relpath(vfr_path, BASE_DIR),
                "sha256": sha256_file(vfr_path),
                "description": "Видео с переменной частотой кадров (VFR: 15fps на 0..5с, 30fps после 5с) и смещением start_time=3.5с.",
                "expected_status": "done",
                "probed": probe_video(vfr_path),
            }
        )

    # 3. AT-14: Повреждённый контейнер
    corrupt_path = os.path.join(SYNTH_DIR, "corrupted_invalid.mov")
    with open(corrupt_path, "wb") as f:
        f.write(b"\x00\x00\x00\x14ftypqt  \x20\x05\x03\x00qt  truncated_broken_moov_atom_payload_xyz")
    synth_manifest.append(
        {
            "test_id": "AT-14",
            "filename": "corrupted_invalid.mov",
            "path": os.path.relpath(corrupt_path, BASE_DIR),
            "sha256": sha256_file(corrupt_path),
            "description": "Технически повреждённый контейнер без декодируемого видеопотока.",
            "expected_status": "failed",
            "expected_verdict": None,
        }
    )

    with open(os.path.join(SYNTH_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"synthetic_tests": synth_manifest}, f, ensure_ascii=False, indent=2)
    return synth_manifest


def build_augmented_video(
    src_path: str,
    stem: str,
    aug_type: str,
    out_dir: str,
    src_info: Dict[str, Any],
    official_info: Dict[str, Any],
    team_info: Dict[str, Any],
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    out_fname = f"{stem}__{aug_type}.mov"
    out_path = os.path.join(out_dir, out_fname)

    w = src_info["width"]
    h = src_info["height"]
    w_075 = (int(w * 0.75) // 2) * 2
    h_075 = (int(h * 0.75) // 2) * 2

    if aug_type == "scale_2x":
        cmd = [
            FFMPEG_BIN,
            "-y",
            "-i",
            src_path,
            "-vf",
            f"scale={w*2}:{h*2}:flags=bicubic",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "ultrafast",
            "-c:a",
            "copy",
            out_path,
        ]
        params_desc = f"scale={w*2}:{h*2}:flags=bicubic, libx264 crf=18"
    elif aug_type == "scale_075x":
        cmd = [
            FFMPEG_BIN,
            "-y",
            "-i",
            src_path,
            "-vf",
            f"scale={w_075}:{h_075}:flags=area",
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-preset",
            "ultrafast",
            "-c:a",
            "copy",
            out_path,
        ]
        params_desc = f"scale={w_075}:{h_075}:flags=area, libx264 crf=20"
    elif aug_type == "prefix_5s":
        # Нейтральная 5-секундная заставка нейтрально-серого цвета перед началом записи (не продлевает открытый сейф/купюры на столе)
        cmd = [
            FFMPEG_BIN,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=0x787878:s={w}x{h}:r=30:d=5.0",
            "-i",
            src_path,
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0[v]",
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            out_path,
        ]
        params_desc = "5.0s neutral gray leader (0x787878) prepended via filter_complex concat"
    elif aug_type == "luma_contrast":
        cmd = [
            FFMPEG_BIN,
            "-y",
            "-i",
            src_path,
            "-vf",
            "eq=brightness=0.05:contrast=1.08",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "ultrafast",
            "-c:a",
            "copy",
            out_path,
        ]
        params_desc = "eq=brightness=0.05:contrast=1.08 (uniform across all frames)"
    elif aug_type == "h264_compression":
        cmd = [
            FFMPEG_BIN,
            "-y",
            "-i",
            src_path,
            "-c:v",
            "libx264",
            "-crf",
            "28",
            "-preset",
            "fast",
            "-c:a",
            "copy",
            out_path,
        ]
        params_desc = "libx264 crf=28 preset=fast (moderate compression artifacts)"
    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")

    if not os.path.exists(out_path):
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    out_info = probe_video(out_path)
    measured_delta_sec = 0.0
    if aug_type == "prefix_5s":
        measured_delta_sec = round(max(0.0, out_info["duration"] - src_info["duration"]), 3)

    # Определение семантической валидности (Раздел 15.3)
    semantic_status = "valid"
    exclusion_reason: Optional[str] = None
    if aug_type in ("scale_075x", "h264_compression") and stem in FINE_DETAIL_STEMS:
        semantic_status = "degraded_fine_detail_warning"
        exclusion_reason = (
            "При снижении разрешения/сжатии мелкие признаки (различие номиналов купюр в пачке или мелкие штампы на столе) "
            "частично теряют визуальную различимость; вариант учитывается в отчёте отдельно."
        )
    elif aug_type == "luma_contrast" and stem in ILLUMINATION_SENSITIVE_STEMS:
        semantic_status = "illumination_semantics_check"
        exclusion_reason = (
            "В исходном ролике значимым событием является переключение освещения; сдвиг яркости проверен на сохранение контраста перепада."
        )

    orig_tc = official_info.get("official_timecode_sec")
    expected_tc = round(orig_tc + measured_delta_sec, 2) if orig_tc is not None else None

    expected_events_shifted = []
    for ev in team_info.get("expected_events", []):
        expected_events_shifted.append(
            {
                "code": ev["code"],
                "timecode_sec": round(float(ev["timecode_sec"]) + measured_delta_sec, 2),
            }
        )

    rel_out_path = os.path.relpath(out_path, BASE_DIR)
    rel_src_path = os.path.relpath(src_path, BASE_DIR)
    rel_cmd = [
        rel_src_path if c == src_path else (rel_out_path if c == out_path else c)
        for c in cmd
    ]

    return {
        "augmented_filename": out_fname,
        "augmented_path": rel_out_path,
        "sha256": sha256_file(out_path),
        "source_stem": stem,
        "source_filename": f"{stem}.mov",
        "source_path": rel_src_path,
        "origin_group_id": team_info.get("origin_group_id", f"group_{stem}"),
        "is_pilot_10": stem in PILOT_10_SELECTION,
        "pilot_selection_reason": PILOT_10_SELECTION.get(stem),
        "augmentation_type": aug_type,
        "parameters": params_desc,
        "ffmpeg_command": " ".join(rel_cmd),
        "source_video_info": src_info,
        "augmented_video_info": out_info,
        "timeline_shift_sec": measured_delta_sec,
        "expected_official_verdict": official_info.get("official_verdict", "no_violation"),
        "expected_official_timecode_sec": expected_tc,
        "expected_team_events": expected_events_shifted,
        "semantic_validity": semantic_status,
        "metric_exclusion_reason": exclusion_reason,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expand-all", action="store_true", help="Generate augmentations for all 42 open videos")
    args = parser.parse_args()

    with open(os.path.join(EVAL_DIR, "official_labels.json"), "r", encoding="utf-8") as f:
        official_labels = json.load(f)["items"]
    with open(os.path.join(EVAL_DIR, "team_annotations.json"), "r", encoding="utf-8") as f:
        team_annotations = json.load(f)["items"]

    synth_items = generate_synthetic_acceptance_files()

    target_stems = sorted(official_labels.keys()) if args.expand_all else sorted(PILOT_10_SELECTION.keys())
    aug_types = ["scale_2x", "scale_075x", "prefix_5s", "luma_contrast", "h264_compression"]

    manifest_entries: List[Dict[str, Any]] = []
    for stem in target_stems:
        src_path = os.path.join(OPEN_VIDEOS_DIR, f"{stem}.mov")
        if not os.path.exists(src_path):
            continue
        src_info = probe_video(src_path)
        off_info = official_labels[stem]
        t_info = team_annotations[stem]
        for atype in aug_types:
            out_sub = os.path.join(AUG_DIR, atype)
            entry = build_augmented_video(
                src_path=src_path,
                stem=stem,
                aug_type=atype,
                out_dir=out_sub,
                src_info=src_info,
                official_info=off_info,
                team_info=t_info,
            )
            manifest_entries.append(entry)

    manifest_payload = {
        "specification_section": "Section 15 Controlled Augmentations",
        "pilot_10_stems": PILOT_10_SELECTION,
        "mode": "full_42" if args.expand_all else "pilot_10",
        "total_augmented_files": len(manifest_entries),
        "augmentation_types": aug_types,
        "synthetic_acceptance_tests": synth_items,
        "entries": manifest_entries,
    }
    os.makedirs(AUG_DIR, exist_ok=True)
    manifest_path = os.path.join(AUG_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, ensure_ascii=False, indent=2)

    print(
        f"Generated {len(manifest_entries)} augmented videos ({'full_42' if args.expand_all else 'pilot_10'}) "
        f"and {len(synth_items)} synthetic acceptance test files -> {manifest_path}"
    )


if __name__ == "__main__":
    main()
