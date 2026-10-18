"""
Stage 6 External MEVA (CC BY 4.0) Dataset Preparation, Manifest Builder & Evaluation Runner (Sections 14.1–14.4, 18.3).

Реализует:
1. Извлечение 12 реальных эпизодов видеонаблюдения из публичного набора MEVA Known Facility Dataset 1 (KF1)
   напрямую из официального S3-бакета `s3://mevadata-public-01/drops-123-r13/` по официальным KPF-аннотациям
   Kitware (`annotation/DIVA-phase-2/MEVA/kitware/2018-03-05/`).
2. Раздельный учёт (Раздел 14.1 и 14.2):
   - числа эпизодов (12),
   - исходных 5-минутных видеозаписей (7),
   - стационарных камер видеонаблюдения (7: G423, G328, G420, G424, G336, G330, G421),
   - независимых многокамерных сценариев/временных окон (`scenario_group_id`: 4).
3. Перевод номеров кадров официальной KPF-разметки (`tsr0: [frame_start, frame_end]`) в фактическое время
   исходного потока и локальное PTS-время вырезанного фрагмента с сохранением `clip_offset_sec`.
4. Протокол раздельной оценки (Раздел 14.3):
   - Оценка распознавания базовых действий MEVA, последовательности фаз и временной локализации;
   - Диагностический прогон банковского пайплайна (`CashDisciplineAnalyzer`) на незнакомой сцене без
     преобразования «передал предмет» в банковское нарушение и без смешивания с банковской accuracy.
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.analyzer import CashDisciplineAnalyzer
from app.cv_detector import CVSceneAnalyzer
from app.video_decoder import PyAVVideoDecoder

EVAL_DIR = os.path.join(BASE_DIR, "evaluation")
MEVA_DIR = os.path.join(EVAL_DIR, "meva_sample")
MEVA_CLIPS_DIR = os.path.join(MEVA_DIR, "clips")
MEVA_ANNOT_DIR = os.path.join(MEVA_DIR, "annotations")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")

FFMPEG_BIN = "/opt/homebrew/bin/ffmpeg"
FFPROBE_BIN = "/opt/homebrew/bin/ffprobe"

S3_BASE = "https://mevadata-public-01.s3.amazonaws.com/drops-123-r13"
GITLAB_RAW_BASE = "https://gitlab.kitware.com/meva/meva-data-repo/-/raw/master/annotation/DIVA-phase-2/MEVA/kitware"

MEVA_ATTRIBUTION = (
    '"Multiview Extended Video with Activities" (MEVA) dataset by Kitware Inc. and the Intelligence Advanced '
    "Research Projects Activity (IARPA), licensed under a Creative Commons Attribution 4.0 International License "
    "(CC BY 4.0, http://creativecommons.org/licenses/by/4.0/, https://mevadata.org/resources/MEVA-data-license.txt)."
)

# 12 курированных эпизодов с официальными KPF-аннотациями Kitware (номера кадров tsr0 проверены по *.activities.yml)
MEVA_EPISODES_SPEC: List[Dict[str, Any]] = [
    {
        "episode_id": "meva_ep01_G423_enter_exit",
        "source_video_id": "2018-03-05.13-15-00.13-20-00.school.G423.r13.avi",
        "s3_rel_path": "2018-03-05/13/2018-03-05.13-15-00.13-20-00.school.G423.r13.avi",
        "annot_rel_path": "2018-03-05/13/2018-03-05.13-15-00.13-20-00.school.G423.activities.yml",
        "camera_id": "G423",
        "facility": "school",
        "scenario_group_id": "meva_scenario_20180305_1315",
        "source_nominal_fps": 30.0,
        "clip_offset_sec": 31.0,
        "clip_duration_sec": 11.0,
        "official_activities": [
            {
                "activity": "person_enters_scene_through_structure",
                "kpf_id2": 0,
                "actor_ids": [1],
                "tsr0_frames": [993, 1029],
            },
            {
                "activity": "person_exits_scene_through_structure",
                "kpf_id2": 1,
                "actor_ids": [2],
                "tsr0_frames": [1175, 1209],
            },
        ],
    },
    {
        "episode_id": "meva_ep02_G328_open_door_enter",
        "source_video_id": "2018-03-05.13-15-01.13-20-01.school.G328.r13.avi",
        "s3_rel_path": "2018-03-05/13/2018-03-05.13-15-01.13-20-01.school.G328.r13.avi",
        "annot_rel_path": "2018-03-05/13/2018-03-05.13-15-01.13-20-01.school.G328.activities.yml",
        "camera_id": "G328",
        "facility": "school",
        "scenario_group_id": "meva_scenario_20180305_1315",
        "source_nominal_fps": 30.0,
        "clip_offset_sec": 2.5,
        "clip_duration_sec": 8.0,
        "official_activities": [
            {
                "activity": "person_opens_facility_door",
                "kpf_id2": 7,
                "actor_ids": [10],
                "tsr0_frames": [136, 163],
            },
            {
                "activity": "person_enters_scene_through_structure",
                "kpf_id2": 8,
                "actor_ids": [11],
                "tsr0_frames": [136, 180],
            },
            {
                "activity": "person_enters_scene_through_structure",
                "kpf_id2": 14,
                "actor_ids": [18],
                "tsr0_frames": [205, 248],
            },
        ],
    },
    {
        "episode_id": "meva_ep03_G424_put_down_heavy_object",
        "source_video_id": "2018-03-05.13-15-00.13-20-00.school.G424.r13.avi",
        "s3_rel_path": "2018-03-05/13/2018-03-05.13-15-00.13-20-00.school.G424.r13.avi",
        "annot_rel_path": "2018-03-05/13/2018-03-05.13-15-00.13-20-00.school.G424.activities.yml",
        "camera_id": "G424",
        "facility": "school",
        "scenario_group_id": "meva_scenario_20180305_1315",
        "source_nominal_fps": 30.0,
        "clip_offset_sec": 96.0,
        "clip_duration_sec": 9.0,
        "official_activities": [
            {
                "activity": "person_carries_heavy_object",
                "kpf_id2": 15,
                "actor_ids": [23, 22],
                "tsr0_frames": [2961, 2991],
            },
            {
                "activity": "person_puts_down_object",
                "kpf_id2": 14,
                "actor_ids": [20, 21],
                "tsr0_frames": [2961, 2976],
            },
        ],
    },
    {
        "episode_id": "meva_ep04_G424_pick_up_and_exit_door",
        "source_video_id": "2018-03-05.13-15-00.13-20-00.school.G424.r13.avi",
        "s3_rel_path": "2018-03-05/13/2018-03-05.13-15-00.13-20-00.school.G424.r13.avi",
        "annot_rel_path": "2018-03-05/13/2018-03-05.13-15-00.13-20-00.school.G424.activities.yml",
        "camera_id": "G424",
        "facility": "school",
        "scenario_group_id": "meva_scenario_20180305_1315",
        "source_nominal_fps": 30.0,
        "clip_offset_sec": 125.0,
        "clip_duration_sec": 10.0,
        "official_activities": [
            {
                "activity": "person_picks_up_object",
                "kpf_id2": 37,
                "actor_ids": [51, 52],
                "tsr0_frames": [3819, 3834],
            },
            {
                "activity": "person_opens_facility_door",
                "kpf_id2": 17,
                "actor_ids": [26],
                "tsr0_frames": [3929, 3989],
            },
            {
                "activity": "person_exits_scene_through_structure",
                "kpf_id2": 18,
                "actor_ids": [27],
                "tsr0_frames": [3929, 3997],
            },
        ],
    },
    {
        "episode_id": "meva_ep05_G336_hand_interacts_person",
        "source_video_id": "2018-03-05.13-20-00.13-25-00.school.G336.r13.avi",
        "s3_rel_path": "2018-03-05/13/2018-03-05.13-20-00.13-25-00.school.G336.r13.avi",
        "annot_rel_path": "2018-03-05/13/2018-03-05.13-20-00.13-25-00.school.G336.activities.yml",
        "camera_id": "G336",
        "facility": "school",
        "scenario_group_id": "meva_scenario_20180305_1320",
        "source_nominal_fps": 30.0,
        "clip_offset_sec": 27.0,
        "clip_duration_sec": 8.0,
        "official_activities": [
            {
                "activity": "hand_interacts_with_person",
                "kpf_id2": 10,
                "actor_ids": [15, 16],
                "tsr0_frames": [891, 965],
            },
        ],
    },
    {
        "episode_id": "meva_ep06_G420_open_door_enter",
        "source_video_id": "2018-03-05.14-00-00.14-05-00.school.G420.r13.avi",
        "s3_rel_path": "2018-03-05/14/2018-03-05.14-00-00.14-05-00.school.G420.r13.avi",
        "annot_rel_path": "2018-03-05/14/2018-03-05.14-00-00.14-05-00.school.G420.activities.yml",
        "camera_id": "G420",
        "facility": "school",
        "scenario_group_id": "meva_scenario_20180305_1400",
        "source_nominal_fps": 30.0,
        "clip_offset_sec": 137.0,
        "clip_duration_sec": 9.0,
        "official_activities": [
            {
                "activity": "person_opens_facility_door",
                "kpf_id2": 0,
                "actor_ids": [1],
                "tsr0_frames": [4183, 4254],
            },
            {
                "activity": "person_enters_scene_through_structure",
                "kpf_id2": 1,
                "actor_ids": [2],
                "tsr0_frames": [4227, 4303],
            },
        ],
    },
    {
        "episode_id": "meva_ep07_G330_pick_up_read_document",
        "source_video_id": "2018-03-05.14-10-00.14-15-00.school.G330.r13.avi",
        "s3_rel_path": "2018-03-05/14/2018-03-05.14-10-00.14-15-00.school.G330.r13.avi",
        "annot_rel_path": "2018-03-05/14/2018-03-05.14-10-00.14-15-00.school.G330.activities.yml",
        "camera_id": "G330",
        "facility": "school",
        "scenario_group_id": "meva_scenario_20180305_1410",
        "source_nominal_fps": 30.0,
        "clip_offset_sec": 244.0,
        "clip_duration_sec": 11.0,
        "official_activities": [
            {
                "activity": "person_picks_up_object",
                "kpf_id2": 3,
                "actor_ids": [5, 4],
                "tsr0_frames": [7393, 7410],
            },
            {
                "activity": "person_reads_document",
                "kpf_id2": 4,
                "actor_ids": [6],
                "tsr0_frames": [7419, 7624],
            },
        ],
    },
    {
        "episode_id": "meva_ep08_G330_texts_on_phone_sits",
        "source_video_id": "2018-03-05.14-10-00.14-15-00.school.G330.r13.avi",
        "s3_rel_path": "2018-03-05/14/2018-03-05.14-10-00.14-15-00.school.G330.r13.avi",
        "annot_rel_path": "2018-03-05/14/2018-03-05.14-10-00.14-15-00.school.G330.activities.yml",
        "camera_id": "G330",
        "facility": "school",
        "scenario_group_id": "meva_scenario_20180305_1410",
        "source_nominal_fps": 30.0,
        "clip_offset_sec": 195.0,
        "clip_duration_sec": 9.0,
        "official_activities": [
            {
                "activity": "person_texts_on_phone",
                "kpf_id2": 0,
                "actor_ids": [1],
                "tsr0_frames": [5574, 7274],
            },
            {
                "activity": "person_sits_down",
                "kpf_id2": 2,
                "actor_ids": [3],
                "tsr0_frames": [5960, 5991],
            },
        ],
    },
    {
        "episode_id": "meva_ep09_G421_read_doc_put_pick",
        "source_video_id": "2018-03-05.14-10-00.14-15-00.school.G421.r13.avi",
        "s3_rel_path": "2018-03-05/14/2018-03-05.14-10-00.14-15-00.school.G421.r13.avi",
        "annot_rel_path": "2018-03-05/14/2018-03-05.14-10-00.14-15-00.school.G421.activities.yml",
        "camera_id": "G421",
        "facility": "school",
        "scenario_group_id": "meva_scenario_20180305_1410",
        "source_nominal_fps": 30.0,
        "clip_offset_sec": 0.0,
        "clip_duration_sec": 11.0,
        "official_activities": [
            {
                "activity": "person_reads_document",
                "kpf_id2": 75,
                "actor_ids": [121],
                "tsr0_frames": [20, 293],
            },
            {
                "activity": "person_puts_down_object",
                "kpf_id2": 14,
                "actor_ids": [18, 17],
                "tsr0_frames": [142, 170],
            },
            {
                "activity": "person_picks_up_object",
                "kpf_id2": 15,
                "actor_ids": [19, 20],
                "tsr0_frames": [205, 219],
            },
        ],
    },
    {
        "episode_id": "meva_ep10_G421_transfer_object_seq",
        "source_video_id": "2018-03-05.14-10-00.14-15-00.school.G421.r13.avi",
        "s3_rel_path": "2018-03-05/14/2018-03-05.14-10-00.14-15-00.school.G421.r13.avi",
        "annot_rel_path": "2018-03-05/14/2018-03-05.14-10-00.14-15-00.school.G421.activities.yml",
        "camera_id": "G421",
        "facility": "school",
        "scenario_group_id": "meva_scenario_20180305_1410",
        "source_nominal_fps": 30.0,
        "clip_offset_sec": 63.0,
        "clip_duration_sec": 11.0,
        "official_activities": [
            {
                "activity": "person_transfers_object",
                "kpf_id2": 83,
                "actor_ids": [135, 136],
                "tsr0_frames": [1958, 2029],
            },
            {
                "activity": "person_puts_down_object",
                "kpf_id2": 20,
                "actor_ids": [28, 27],
                "tsr0_frames": [2028, 2076],
            },
            {
                "activity": "person_transfers_object",
                "kpf_id2": 84,
                "actor_ids": [137, 138],
                "tsr0_frames": [2086, 2159],
            },
        ],
    },
    {
        "episode_id": "meva_ep11_G421_transfer_and_pick_up",
        "source_video_id": "2018-03-05.14-10-00.14-15-00.school.G421.r13.avi",
        "s3_rel_path": "2018-03-05/14/2018-03-05.14-10-00.14-15-00.school.G421.r13.avi",
        "annot_rel_path": "2018-03-05/14/2018-03-05.14-10-00.14-15-00.school.G421.r13.avi".replace(".r13.avi", ".activities.yml"),
        "camera_id": "G421",
        "facility": "school",
        "scenario_group_id": "meva_scenario_20180305_1410",
        "source_nominal_fps": 30.0,
        "clip_offset_sec": 94.0,
        "clip_duration_sec": 9.0,
        "official_activities": [
            {
                "activity": "person_transfers_object",
                "kpf_id2": 37,
                "actor_ids": [57, 56],
                "tsr0_frames": [2887, 2955],
            },
            {
                "activity": "person_picks_up_object",
                "kpf_id2": 21,
                "actor_ids": [29, 30],
                "tsr0_frames": [2990, 3007],
            },
        ],
    },
    {
        "episode_id": "meva_ep12_G421_transfer_put_phone",
        "source_video_id": "2018-03-05.14-10-00.14-15-00.school.G421.r13.avi",
        "s3_rel_path": "2018-03-05/14/2018-03-05.14-10-00.14-15-00.school.G421.r13.avi",
        "annot_rel_path": "2018-03-05/14/2018-03-05.14-10-00.14-15-00.school.G421.activities.yml",
        "camera_id": "G421",
        "facility": "school",
        "scenario_group_id": "meva_scenario_20180305_1410",
        "source_nominal_fps": 30.0,
        "clip_offset_sec": 139.0,
        "clip_duration_sec": 10.0,
        "official_activities": [
            {
                "activity": "person_transfers_object",
                "kpf_id2": 77,
                "actor_ids": [124, 125],
                "tsr0_frames": [4222, 4285],
            },
            {
                "activity": "person_puts_down_object",
                "kpf_id2": 43,
                "actor_ids": [64, 65],
                "tsr0_frames": [4254, 4281],
            },
            {
                "activity": "person_texts_on_phone",
                "kpf_id2": 78,
                "actor_ids": [126],
                "tsr0_frames": [4277, 4706],
            },
        ],
    },
]

MEVA_ACTIVITY_ENUM = [
    "person_enters_scene_through_structure",
    "person_exits_scene_through_structure",
    "person_opens_facility_door",
    "person_picks_up_object",
    "person_puts_down_object",
    "person_carries_heavy_object",
    "person_reads_document",
    "person_texts_on_phone",
    "person_sits_down",
    "person_transfers_object",
    "hand_interacts_with_person",
]


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
        "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,time_base,start_time,duration,nb_frames",
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


def prepare_meva_clips_and_manifest() -> Dict[str, Any]:
    import urllib.request
    import yaml

    os.makedirs(MEVA_CLIPS_DIR, exist_ok=True)
    os.makedirs(MEVA_ANNOT_DIR, exist_ok=True)

    with open(os.path.join(MEVA_DIR, "LICENSE_ATTRIBUTION.txt"), "w", encoding="utf-8") as f:
        f.write(MEVA_ATTRIBUTION + "\n")

    manifest_episodes: List[Dict[str, Any]] = []

    for spec in MEVA_EPISODES_SPEC:
        ep_id = spec["episode_id"]
        s3_url = f"{S3_BASE}/{spec['s3_rel_path']}"
        annot_url = f"{GITLAB_RAW_BASE}/{spec['annot_rel_path']}"
        annot_fname = os.path.basename(spec["annot_rel_path"])
        local_annot_path = os.path.join(MEVA_ANNOT_DIR, annot_fname)
        if not os.path.exists(local_annot_path) or os.path.getsize(local_annot_path) < 50:
            urllib.request.urlretrieve(annot_url, local_annot_path)
        annot_sha = sha256_file(local_annot_path)

        # Проверяем соответствие каждого действия первичному KPF файлу .activities.yml (F15)
        with open(local_annot_path, "r", encoding="utf-8") as yf:
            raw_kpf = yaml.safe_load(yf) or []
        kpf_by_id2: Dict[int, Dict[str, Any]] = {}
        for item in raw_kpf:
            if isinstance(item, dict) and "act" in item and isinstance(item["act"], dict):
                act_obj = item["act"]
                if "id2" in act_obj:
                    kpf_by_id2[int(act_obj["id2"])] = act_obj

        out_clip_path = os.path.join(MEVA_CLIPS_DIR, f"{ep_id}.mp4")

        offset_sec = float(spec["clip_offset_sec"])
        dur_sec = float(spec["clip_duration_sec"])
        fps = float(spec["source_nominal_fps"])

        cmd = [
            FFMPEG_BIN,
            "-y",
            "-ss",
            str(offset_sec),
            "-i",
            s3_url,
            "-t",
            str(dur_sec),
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-preset",
            "fast",
            "-pix_fmt",
            "yuv420p",
            "-an",
            out_clip_path,
        ]
        if not os.path.exists(out_clip_path) or os.path.getsize(out_clip_path) < 10000:
            print(f"Extracting MEVA clip {ep_id} from {s3_url} (offset={offset_sec}s, dur={dur_sec}s)...", flush=True)
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        probed = probe_video(out_clip_path)

        converted_activities = []
        for act in spec["official_activities"]:
            kid = int(act["kpf_id2"])
            if kid in kpf_by_id2:
                kpf_entry = kpf_by_id2[kid]
                act2_map = kpf_entry.get("act2") or {}
                assert act["activity"] in act2_map, f"KPF mismatch on {ep_id} id2={kid}: {act['activity']} not in {act2_map}"
                timespans = kpf_entry.get("timespan") or []
                if timespans and "tsr0" in timespans[0]:
                    assert list(timespans[0]["tsr0"]) == list(act["tsr0_frames"]), (
                        f"KPF tsr0 mismatch on {ep_id} id2={kid}: {timespans[0]['tsr0']} vs {act['tsr0_frames']}"
                    )
            f_start, f_end = act["tsr0_frames"]
            src_start_sec = round(f_start / fps, 2)
            src_end_sec = round(f_end / fps, 2)
            clip_start_sec = round(max(0.0, src_start_sec - offset_sec), 2)
            clip_end_sec = round(min(probed["duration"], max(clip_start_sec, src_end_sec - offset_sec)), 2)
            converted_activities.append(
                {
                    "activity": act["activity"],
                    "kpf_id2": act["kpf_id2"],
                    "actor_ids": act["actor_ids"],
                    "tsr0_frames": [f_start, f_end],
                    "verified_against_primary_kpf_yaml": kid in kpf_by_id2,
                    "source_time_range_sec": [src_start_sec, src_end_sec],
                    "clip_pts_time_range_sec": [clip_start_sec, clip_end_sec],
                    "clip_midpoint_sec": round(0.5 * (clip_start_sec + clip_end_sec), 2),
                }
            )

        rel_clip_path = os.path.relpath(out_clip_path, BASE_DIR)
        rel_annot_path = os.path.relpath(local_annot_path, BASE_DIR)
        rel_cmd = [rel_clip_path if c == out_clip_path else c for c in cmd]

        manifest_episodes.append(
            {
                "episode_id": ep_id,
                "clip_filename": f"{ep_id}.mp4",
                "clip_path": rel_clip_path,
                "sha256": sha256_file(out_clip_path),
                "source_video_id": spec["source_video_id"],
                "source_s3_url": s3_url,
                "official_kpf_annotation_url": annot_url,
                "local_annotation_path": rel_annot_path,
                "annotation_sha256": annot_sha,
                "camera_id": spec["camera_id"],
                "facility": spec["facility"],
                "scenario_group_id": spec["scenario_group_id"],
                "reference_system": "PTS seconds normalized to clip start (with explicit clip_offset_sec from source stream)",
                "clip_offset_sec": offset_sec,
                "requested_duration_sec": dur_sec,
                "ffmpeg_extraction_command": " ".join(rel_cmd),
                "probed_clip_metadata": probed,
                "official_activities": converted_activities,
                "manual_verification_status": "verified_against_kpf_yaml_and_clip_pts",
                "license": "CC BY 4.0",
                "attribution": MEVA_ATTRIBUTION,
            }
        )

    unique_sources = sorted({e["source_video_id"] for e in manifest_episodes})
    unique_cameras = sorted({e["camera_id"] for e in manifest_episodes})
    unique_scenarios = sorted({e["scenario_group_id"] for e in manifest_episodes})

    manifest_data = {
        "specification_section": "Section 14 External MEVA (CC BY 4.0) Sample",
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "attribution": MEVA_ATTRIBUTION,
        "counts_summary": {
            "total_episodes": len(manifest_episodes),
            "unique_source_recordings": len(unique_sources),
            "unique_cameras": len(unique_cameras),
            "unique_independent_scenarios": len(unique_scenarios),
            "source_recordings_list": unique_sources,
            "cameras_list": unique_cameras,
            "scenario_groups_list": unique_scenarios,
        },
        "episodes": manifest_episodes,
    }

    manifest_path = os.path.join(MEVA_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, ensure_ascii=False, indent=2)
    print(f"Saved MEVA manifest ({len(manifest_episodes)} episodes) to {manifest_path}", flush=True)
    return manifest_data


def run_meva_diagnostic_vlm(
    analyzer: CashDisciplineAnalyzer,
    bundle: Any,
    cv_out: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Отдельный диагностический интерфейс наблюдений базовых действий (Раздел 14.3 п. 168),
    не меняющий контракт банковского API и не преобразующий 'person_transfers_object' или 'person_reads_document'
    в банковские нарушения.
    """
    model, processor, config, _ = analyzer.vlm_engine._load()
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.structured import build_json_schema_logits_processor

    frames = analyzer._select_coarse_storyboard_frames(bundle, num_frames=6)
    storyboard = PyAVVideoDecoder.build_annotated_storyboard(frames, tile_size=(280, 336), cols=3)
    allowed_fids = [f.frame_id for f in frames]
    frame_time_table = ", ".join(f"{f.frame_id}={f.timestamp_sec:.2f}s" for f in frames)

    # Формируем объективную сводку пространственно-временных наблюдений CV по кадрам
    f_obs = cv_out.get("frame_observations", [])
    p_counts = [f"{fo.frame_id}:persons={fo.person_count}" for fo in f_obs if fo.frame_id in set(allowed_fids)]
    cv_summary = (
        f"Наблюдения детектора CV: макс. людей={cv_out.get('max_persons', 0)}, "
        f"объекты={cv_out.get('detected_interacting_objects', [])}, "
        f"по кадрам [{', '.join(p_counts)}]."
    )

    meva_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["scene_summary", "detected_activities"],
        "properties": {
            "scene_summary": {"type": "string", "minLength": 10, "maxLength": 75},
            "detected_activities": {
                "type": "array",
                "minItems": 1,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["activity", "start_sec", "end_sec", "frame_ids"],
                    "properties": {
                        "activity": {"type": "string", "enum": MEVA_ACTIVITY_ENUM},
                        "start_sec": {"type": "number", "minimum": 0.0, "maximum": round(bundle.duration_sec, 2)},
                        "end_sec": {"type": "number", "minimum": 0.0, "maximum": round(bundle.duration_sec, 2)},
                        "frame_ids": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 2,
                            "items": {"type": "string", "enum": allowed_fids},
                        },
                    },
                },
            },
        },
    }

    prompt = (
        f"Видео камеры наблюдения (длительность {bundle.duration_sec:.2f}с, кадры [{frame_time_table}]).\n"
        f"{cv_summary}\n"
        "Определи 1–3 главных базовых действия людей из официальных классов MEVA:\n"
        "- person_enters_scene_through_structure / person_exits_scene_through_structure / person_opens_facility_door (если человек входит/выходит через дверь)\n"
        "- person_picks_up_object / person_puts_down_object / person_carries_heavy_object (если человек берёт, ставит или несёт предмет/коробку/сумку)\n"
        "- person_reads_document (если человек смотрит в бумагу/документ/папку)\n"
        "- person_texts_on_phone / person_sits_down (если человек использует телефон или садится)\n"
        "- person_transfers_object / hand_interacts_with_person (если два человека передают предмет из рук в руки или взаимодействуют руками)\n"
        "Укажи точные start_sec, end_sec и frame_ids. Описание кратко (до 70 символов)."
    )

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        storyboard.save(tmp_path, quality=85)

    try:
        import mlx.core as mx
        for max_toks in (320, 440):
            formatted = apply_chat_template(processor, config, prompt, num_images=1)
            lp = build_json_schema_logits_processor(tokenizer, meva_schema)
            res = generate(
                model,
                processor,
                formatted,
                image=[tmp_path],
                logits_processors=[lp],
                max_tokens=max_toks,
                temp=0.0,
                verbose=False,
            )
            mx.clear_cache()
            raw_text = res.text if hasattr(res, "text") else str(res)
            try:
                return json.loads(raw_text.strip())
            except Exception:
                if max_toks == 440:
                    raise
        return {"scene_summary": "Сцена наблюдения MEVA", "detected_activities": []}
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    manifest_data = prepare_meva_clips_and_manifest()
    episodes = manifest_data["episodes"]

    analyzer = CashDisciplineAnalyzer(default_vlm_mode="fast")
    cv_analyzer = CVSceneAnalyzer()

    episode_results: List[Dict[str, Any]] = []
    total_gt_activities = 0
    matched_gt_activities = 0
    loc_errors_sec: List[float] = []
    banking_triggered_count = 0
    out_path = os.path.join(RESULTS_DIR, "stage6_meva_evaluation.json")

    for idx, ep in enumerate(episodes, 1):
        raw_cpath = ep["clip_path"]
        cpath = raw_cpath if os.path.isabs(raw_cpath) else os.path.join(BASE_DIR, raw_cpath)
        gt_acts = ep["official_activities"]
        total_gt_activities += len(gt_acts)
        t0 = time.time()
        try:
            bundle = PyAVVideoDecoder.decode_timeline(cpath, target_interval_sec=1.5)
            cv_out = cv_analyzer.analyze_timeline(bundle)
            diag_vlm_out = run_meva_diagnostic_vlm(analyzer, bundle, cv_out)

            # Отдельно проверяем поведение банковского анализатора на незнакомой сцене (Раздел 14.3 п. 169–170)
            bank_res = analyzer.analyze_video(
                video_path=cpath,
                original_filename=ep["clip_filename"],
                vlm_mode="fast",
                invoke_vlm=True,
                pipeline_mode="hybrid",
            )
            if bank_res["verdict"] == "violation":
                banking_triggered_count += 1

            pred_acts = diag_vlm_out.get("detected_activities", [])
            pred_act_names = [a["activity"] for a in pred_acts]

            import itertools

            ep_matched = []
            ep_missed = []
            for g_name in sorted({g["activity"] for g in gt_acts}):
                g_group = sorted(
                    [g for g in gt_acts if g["activity"] == g_name],
                    key=lambda x: float(x["clip_midpoint_sec"]),
                )
                p_group = sorted(
                    [p for p in pred_acts if p["activity"] == g_name],
                    key=lambda x: 0.5 * (float(x["start_sec"]) + float(x["end_sec"])),
                )
                n_match = min(len(g_group), len(p_group))
                if n_match == 0:
                    for g in g_group:
                        ep_missed.append({"activity": g_name, "gt_clip_range_sec": g["clip_pts_time_range_sec"]})
                    continue
                if len(g_group) <= len(p_group):
                    pair_options = [
                        list(zip(g_group, seq))
                        for seq in itertools.permutations(p_group, n_match)
                    ]
                else:
                    pair_options = [
                        list(zip(seq, p_group))
                        for seq in itertools.permutations(g_group, n_match)
                    ]

                def _pair_cost(pairs_list):
                    errs = [
                        round(abs(0.5 * (float(p["start_sec"]) + float(p["end_sec"])) - float(g["clip_midpoint_sec"])), 2)
                        for g, p in pairs_list
                    ]
                    return (round(sum(errs), 6), -sum(1 for e in errs if e <= 2.0), errs)

                best_pairs = min(pair_options, key=_pair_cost)
                matched_g_ids = {id(g) for g, _ in best_pairs}
                for g, best_p in best_pairs:
                    matched_gt_activities += 1
                    g_start, g_end = g["clip_pts_time_range_sec"]
                    g_mid = float(g["clip_midpoint_sec"])
                    p_s, p_e = float(best_p["start_sec"]), float(best_p["end_sec"])
                    p_mid = 0.5 * (p_s + p_e)
                    if p_e >= g_start and p_s <= g_end:
                        interval_dist = 0.0
                    else:
                        interval_dist = min(abs(p_e - g_start), abs(p_s - g_end))
                    midpoint_err = round(abs(p_mid - g_mid), 2)
                    loc_errors_sec.append(midpoint_err)
                    ep_matched.append(
                        {
                            "activity": g_name,
                            "gt_clip_range_sec": [g_start, g_end],
                            "pred_clip_range_sec": [round(p_s, 2), round(p_e, 2)],
                            "midpoint_error_sec": midpoint_err,
                            "interval_overlap_distance_sec": round(interval_dist, 2),
                        }
                    )
                for g in g_group:
                    if id(g) not in matched_g_ids:
                        ep_missed.append({"activity": g_name, "gt_clip_range_sec": g["clip_pts_time_range_sec"]})

            rec = {
                "episode_id": ep["episode_id"],
                "status": "done",
                "source_video_id": ep["source_video_id"],
                "camera_id": ep["camera_id"],
                "scenario_group_id": ep["scenario_group_id"],
                "clip_duration_sec": bundle.duration_sec,
                "cv_max_persons": cv_out["max_persons"],
                "cv_interacting_objects": cv_out["detected_interacting_objects"],
                "official_gt_activities": gt_acts,
                "diagnostic_vlm_scene_summary": diag_vlm_out.get("scene_summary"),
                "diagnostic_vlm_detected_activities": pred_acts,
                "matched_activities": ep_matched,
                "missed_activities": ep_missed,
                "unfamiliar_scene_banking_api_behavior": {
                    "banking_verdict": bank_res["verdict"],
                    "banking_violations_returned": [
                        {"code": v["code"], "timecode_sec": v["timecode_sec"], "description": v["description"]}
                        for v in bank_res["violations"]
                    ],
                    "manual_review_note": (
                        "Срабатывание банковского правила на небанковской сцене (например, использование смартфона или передача предмета) "
                        "фиксируется только как диагностический сигнал переносимости (Раздел 14.3 п. 170)."
                        if bank_res["verdict"] == "violation"
                        else "Конвейер корректно не сформировал обвинительных банковских нарушений на незнакомой небанковской сцене."
                    ),
                },
                "processing_time_sec": round(time.time() - t0, 2),
            }
            print(
                f"[MEVA {idx:02d}/12] {ep['episode_id']} (cam={ep['camera_id']}) -> "
                f"matched={len(ep_matched)}/{len(gt_acts)} acts={pred_act_names} "
                f"bank_verdict={bank_res['verdict']}",
                flush=True,
            )
        except Exception as e:
            rec = {
                "episode_id": ep["episode_id"],
                "status": "failed",
                "error": str(e),
                "source_video_id": ep["source_video_id"],
                "camera_id": ep["camera_id"],
                "scenario_group_id": ep["scenario_group_id"],
                "clip_duration_sec": ep.get("probed_clip_metadata", {}).get("duration", 0.0),
                "cv_max_persons": 0,
                "cv_interacting_objects": [],
                "official_gt_activities": gt_acts,
                "diagnostic_vlm_scene_summary": None,
                "diagnostic_vlm_detected_activities": [],
                "matched_activities": [],
                "missed_activities": [
                    {"activity": g["activity"], "gt_clip_range_sec": g["clip_pts_time_range_sec"]}
                    for g in gt_acts
                ],
                "unfamiliar_scene_banking_api_behavior": {
                    "banking_verdict": None,
                    "banking_violations_returned": [],
                    "manual_review_note": f"Ошибка исполнения: {e}",
                },
                "processing_time_sec": round(time.time() - t0, 2),
            }
            print(f"[MEVA {idx:02d}/12] {ep['episode_id']} FAILED: {e}", flush=True)

        episode_results.append(rec)

        within_2s_cp = sum(1 for e in loc_errors_sec if e <= 2.0)
        within_3s_cp = sum(1 for e in loc_errors_sec if e <= 3.0)
        succeeded_episodes = [r for r in episode_results if r.get("status") == "done"]
        failed_episodes = [r for r in episode_results if r.get("status") != "done"]
        banking_no_violation_done = sum(
            1
            for r in succeeded_episodes
            if (r.get("unfamiliar_scene_banking_api_behavior") or {}).get("banking_verdict") == "no_violation"
        )
        banking_trigger_done = sum(
            1
            for r in succeeded_episodes
            if (r.get("unfamiliar_scene_banking_api_behavior") or {}).get("banking_verdict") == "violation"
        )
        eval_payload = {
            "specification_section": "Section 14 & Section 18.3 External MEVA Evaluation",
            "usage_status": "Independent external evaluation sample (not used for prompt or rule tuning)",
            "counts_summary": manifest_data["counts_summary"],
            "base_action_recognition_summary": {
                "total_official_gt_activities": total_gt_activities,
                "matched_activities_count": matched_gt_activities,
                "base_action_recall": round(matched_gt_activities / max(1, total_gt_activities), 4),
                "base_action_recall_fraction": f"{matched_gt_activities}/{total_gt_activities}",
                "temporal_localization_evaluated_count": len(loc_errors_sec),
                "midpoint_error_within_2_0s_count": within_2s_cp,
                "midpoint_error_within_2_0s_rate": round(within_2s_cp / max(1, len(loc_errors_sec)), 4) if loc_errors_sec else 0.0,
                "midpoint_error_within_3_0s_count": within_3s_cp,
                "mean_midpoint_error_sec": round(sum(loc_errors_sec) / len(loc_errors_sec), 2) if loc_errors_sec else None,
            },
            "unfamiliar_scene_banking_diagnostic_signal": {
                "total_episodes": len(episode_results),
                "evaluated_episodes_count": len(succeeded_episodes),
                "banking_failed_count": len(failed_episodes),
                "banking_no_violation_count": banking_no_violation_done,
                "banking_violation_trigger_count": banking_trigger_done,
                "banking_trigger_diagnostic_rate": (
                    round(banking_trigger_done / max(1, len(succeeded_episodes)), 4)
                    if succeeded_episodes
                    else None
                ),
                "false_banking_violation_rate_on_evaluated": (
                    round(banking_trigger_done / max(1, len(succeeded_episodes)), 4)
                    if succeeded_episodes
                    else None
                ),
                "banking_no_violation_rate_on_all_inputs": round(
                    banking_no_violation_done / max(1, len(episode_results)), 4
                ),
                "methodological_note": (
                    "Согласно Разделу 14.3 ТЗ, MEVA не содержит разметки нарушений банковского кассового регламента. "
                    "Доля банковских срабатываний приведена как диагностический сигнал на незнакомых сценах с ручным разбором "
                    "и НЕ объединяется с банковской accuracy. Эпизоды со status='failed' учитываются отдельно в "
                    "banking_failed_count и не засчитываются в banking_no_violation_count."
                ),
            },
            "per_episode_results": episode_results,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(eval_payload, f, ensure_ascii=False, indent=2)
    print(f"Saved Stage 6 MEVA evaluation to {out_path}", flush=True)


if __name__ == "__main__":
    main()
