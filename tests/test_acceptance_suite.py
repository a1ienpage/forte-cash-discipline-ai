"""
Приёмочный набор автоматизированных тестов pytest (AT-01 .. AT-21, Раздел 19 ТЗ на доработку).
"""

import ast
import base64
import glob
import io
import json
import math
import os
import shutil
import tempfile
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from app.analyzer import CashDisciplineAnalyzer
from app.cv_detector import CVSceneAnalyzer, _person_object_interaction_distance
from app.main import API_KEY, app
from app.regulations import (
    VIOLATION_CATALOG,
    format_violation_entry,
    get_disabled_catalog,
    get_enabled_catalog,
    validate_rule_applicability,
)
from app.schemas import AnalysisResultModel
from app.video_decoder import PyAVVideoDecoder, VideoDecodeError
from app.vlm_engine import LocalVLMEngine, ModelNotCachedError
from app.worker import ProcessJobManager

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYNTH_DIR = os.path.join(BASE_DIR, "evaluation", "synthetic_tests")
AUG_DIR = os.path.join(BASE_DIR, "evaluation", "augmentations")
OPEN_VIDEOS_DIR = os.path.join(BASE_DIR, "open", "videos")


# --- AT-01: Однотонное корректное видео ---
def test_at01_monotone_valid_video():
    mono_path = os.path.join(SYNTH_DIR, "monotone_valid.mov")
    assert os.path.exists(mono_path), "Синтетический файл monotone_valid.mov должен существовать"
    an = CashDisciplineAnalyzer()
    res = an.analyze_video(mono_path, invoke_vlm=True)
    assert res["verdict"] == "no_violation"
    assert res["violations"] == []
    assert res["meta"]["duration_sec"] >= 5.5


# --- AT-02: Обычное корректное поведение ---
def test_at02_normal_clean_behavior():
    clean_path = os.path.join(OPEN_VIDEOS_DIR, "Х-14.mov")
    an = CashDisciplineAnalyzer()
    res = an.analyze_video(clean_path, invoke_vlm=False, pipeline_mode="cv_only")
    assert res["verdict"] == "no_violation"
    assert res["violations"] == []


# --- AT-03: Короткая длительность не подавляет анализ принудительно ---
def test_at03_short_duration_not_blocked():
    import av
    import numpy as np
    from unittest.mock import patch

    # 1. Проверяем в коде app/, что нигде нет подавления нарушений по порогу короткой длительности
    for py_path in glob.glob(os.path.join(BASE_DIR, "app", "*.py")):
        src = open(py_path, "r", encoding="utf-8").read()
        assert "duration < 18" not in src
        assert "duration > 45" not in src
        assert "scene_layout" not in src

    # 2. Реальный прогон 7-секундного клипа (< 10 с) с детекцией personal_phone_usage в cv_only
    with tempfile.TemporaryDirectory() as td:
        short_clip = os.path.join(td, "short_7s.mp4")
        c = av.open(short_clip, mode="w")
        s = c.add_stream("libx264", rate=10)
        s.width, s.height, s.pix_fmt = 640, 360, "yuv420p"
        for _ in range(70):
            f = av.VideoFrame.from_ndarray(np.zeros((360, 640, 3), dtype=np.uint8), format="bgr24")
            for p in s.encode(f):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
        c.close()

        class _Arr(list):
            def tolist(self):
                return list(self)

        class _Box:
            def __init__(self, c, cf, xy):
                self.cls = np.array([c])
                self.conf = np.array([cf])
                self.xyxy = [_Arr(xy)]

        class _Res:
            boxes = [_Box(0, 0.95, [120.0, 50.0, 320.0, 330.0]), _Box(67, 0.90, [200.0, 140.0, 240.0, 190.0])]

        class _FakeYOLO:
            def predict(self, imgs, *a, **kw):
                return [_Res() for _ in imgs]

        an = CashDisciplineAnalyzer()
        an.cv_analyzer.yolo_model = _FakeYOLO()
        out = an.analyze_video(short_clip, invoke_vlm=False, pipeline_mode="cv_only")
        assert 6.5 <= out["meta"]["duration_sec"] <= 7.5
        assert out["verdict"] == "violation"
        assert any(v["code"] == "personal_phone_usage" for v in out["violations"])


# --- AT-04: Переименование и перемещение файла ---
def test_at04_rename_and_move_invariance():
    src_video = os.path.join(OPEN_VIDEOS_DIR, "Х-26.mov")
    an = CashDisciplineAnalyzer()
    res_orig = an.analyze_video(src_video, original_filename="Х-26.mov", invoke_vlm=False, pipeline_mode="cv_only")

    with tempfile.TemporaryDirectory() as tmpdir:
        moved_path = os.path.join(tmpdir, "completely_renamed_999_xyz.mov")
        shutil.copy2(src_video, moved_path)
        res_moved = an.analyze_video(
            moved_path,
            original_filename="completely_renamed_999_xyz.mov",
            invoke_vlm=False,
            pipeline_mode="cv_only",
        )

    assert res_orig["verdict"] == res_moved["verdict"]
    orig_codes = [(v["code"], v["timecode_sec"]) for v in res_orig["violations"]]
    moved_codes = [(v["code"], v["timecode_sec"]) for v in res_moved["violations"]]
    assert orig_codes == moved_codes


# --- AT-05: Масштабирование разрешения ×2 и ×0.75 ---
def test_at05_resolution_scaling_invariance():
    p_orig = os.path.join(OPEN_VIDEOS_DIR, "Х-26.mov")
    p_2x = os.path.join(AUG_DIR, "scale_2x", "Х-26__scale_2x.mov")
    p_075x = os.path.join(AUG_DIR, "scale_075x", "Х-26__scale_075x.mov")
    assert os.path.exists(p_2x) and os.path.exists(p_075x)

    an = CashDisciplineAnalyzer()
    r_orig = an.analyze_video(p_orig, invoke_vlm=False, pipeline_mode="cv_only")
    r_2x = an.analyze_video(p_2x, invoke_vlm=False, pipeline_mode="cv_only")
    r_075x = an.analyze_video(p_075x, invoke_vlm=False, pipeline_mode="cv_only")

    assert r_orig["verdict"] == r_2x["verdict"] == r_075x["verdict"] == "violation"
    t_orig = r_orig["violations"][0]["timecode_sec"]
    t_2x = r_2x["violations"][0]["timecode_sec"]
    t_075x = r_075x["violations"][0]["timecode_sec"]
    assert abs(t_orig - t_2x) <= 1.0
    assert abs(t_orig - t_075x) <= 1.0


# --- AT-06: Нейтральное начало +5 секунд (сдвиг временной шкалы t_new = t_orig + delta) ---
def test_at06_neutral_prefix_5s_shift():
    p_orig = os.path.join(OPEN_VIDEOS_DIR, "Х-26.mov")
    p_pref = os.path.join(AUG_DIR, "prefix_5s", "Х-26__prefix_5s.mov")
    assert os.path.exists(p_pref)

    an = CashDisciplineAnalyzer()
    r_orig = an.analyze_video(p_orig, invoke_vlm=False, pipeline_mode="cv_only")
    r_pref = an.analyze_video(p_pref, invoke_vlm=False, pipeline_mode="cv_only")

    measured_delta = r_pref["meta"]["duration_sec"] - r_orig["meta"]["duration_sec"]
    assert 4.8 <= measured_delta <= 5.2

    t_orig = r_orig["violations"][0]["timecode_sec"]
    t_pref = r_pref["violations"][0]["timecode_sec"]
    assert abs(t_pref - (t_orig + measured_delta)) <= 1.5


# --- AT-07: Умеренные изменения освещения и сжатия + манифест семантической валидности ---
def test_at07_lighting_and_compression_manifest():
    manifest_path = os.path.join(AUG_DIR, "manifest.json")
    assert os.path.exists(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["total_augmented_files"] >= 50
    for entry in manifest["entries"]:
        assert "sha256" in entry
        assert "origin_group_id" in entry
        assert "ffmpeg_command" in entry
        assert "semantic_validity" in entry

    p_luma = os.path.join(AUG_DIR, "luma_contrast", "Х-26__luma_contrast.mov")
    p_comp = os.path.join(AUG_DIR, "h264_compression", "Х-26__h264_compression.mov")
    an = CashDisciplineAnalyzer()
    r_luma = an.analyze_video(p_luma, invoke_vlm=False, pipeline_mode="cv_only")
    r_comp = an.analyze_video(p_comp, invoke_vlm=False, pipeline_mode="cv_only")
    assert r_luma["verdict"] == r_comp["verdict"] == "violation"


# --- AT-08: Переменная частота кадров (VFR) и смещение начала потока (PTS * time_base) ---
def test_at08_vfr_and_stream_start_offset_pts():
    from fractions import Fraction
    import av
    import numpy as np
    from unittest.mock import patch

    vfr_path = os.path.join(SYNTH_DIR, "vfr_offset_test.mov")
    assert os.path.exists(vfr_path)
    bundle = PyAVVideoDecoder.decode_timeline(vfr_path, target_interval_sec=1.0)
    assert bundle.is_vfr_detected is True
    assert abs(bundle.start_offset_sec - 3.5) < 0.1
    assert "normalized_by_stream_start_pts" in bundle.offset_normalization_rule
    first_df = bundle.coarse_frames[0]
    assert abs(first_df.raw_pts_sec - 3.5) < 0.1
    assert first_df.timestamp_sec == 0.0

    # Проверка R08: поток без PTS (pts=None) выставляет pts_fallback_used=True либо вызывает VideoDecodeError при strict_pts=True
    class _NoPtsFrame:
        pts = None
        dts = None
        width = 64
        height = 36
        def to_ndarray(self, format="bgr24"):
            return np.zeros((36, 64, 3), dtype=np.uint8)

    class _NoPtsStream:
        width = 64
        height = 36
        average_rate = Fraction(30, 1)
        base_rate = Fraction(30, 1)
        time_base = Fraction(1, 30)
        start_time = None
        duration = None
        frames = 10
        thread_type = None
        codec_context = None

    class _NoPtsContainer:
        def __init__(self):
            self.streams = type("S", (), {"video": [_NoPtsStream()]})()
            self.duration = None
        def decode(self, video=0):
            for _ in range(10):
                yield _NoPtsFrame()
        def close(self):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    with patch("app.video_decoder.av.open", return_value=_NoPtsContainer()):
        fb_bundle = PyAVVideoDecoder.decode_timeline(vfr_path, target_interval_sec=0.1, strict_pts=False)
        assert fb_bundle.pts_fallback_used is True
        assert fb_bundle.missing_pts_frames_count == 10
        with pytest.raises(VideoDecodeError):
            PyAVVideoDecoder.decode_timeline(vfr_path, target_interval_sec=0.1, strict_pts=True)

    # Проверка R08 (mixed PTS + start_time=300 -> 10.0s): пропуск PTS в кадре #0 или #30 сохраняет единую монотонную шкалу 0..1.9667с
    for missing_set in ({0}, {30}):
        class _MixedStream:
            time_base = Fraction(1, 30)
            average_rate = Fraction(30, 1)
            base_rate = Fraction(30, 1)
            start_time = 300
            duration = 60
            thread_type = None
            codec_context = type("C", (), {"name": "stub"})()

        class _MixedContainer:
            streams = type("S", (), {"video": [_MixedStream()]})()
            duration = 2_000_000
            def decode(self, **kwargs):
                for idx in range(60):
                    fr = av.VideoFrame.from_ndarray(np.zeros((48, 64, 3), np.uint8), format="bgr24")
                    fr.pts = None if idx in missing_set else 300 + idx
                    yield fr
            def close(self):
                pass

        with patch("app.video_decoder.av.open", side_effect=lambda *a, **kw: _MixedContainer()):
            mb = PyAVVideoDecoder.decode_timeline(vfr_path, strict_pts=False)
            assert mb.start_offset_sec == 10.0
            assert mb.duration_sec == 2.0
            assert mb.pts_fallback_used is True
            for i in (0, 1, 29, 30, 31, 59):
                assert mb.frame_pts_map[f"F{i:04d}"]["timestamp_sec"] == round(i / 30, 4)
            assert [f.timestamp_sec for f in mb.coarse_frames] == [0.0, 1.0, 1.9667]
            by_ids = PyAVVideoDecoder.decode_frames_by_ids(
                vfr_path, ["F0000", "F0030"], start_offset_sec=10.0, nominal_fps=30.0
            )
            assert by_ids["F0000"].timestamp_sec == 0.0
            assert by_ids["F0030"].timestamp_sec == 1.0
            with pytest.raises(VideoDecodeError):
                PyAVVideoDecoder.decode_timeline(vfr_path, strict_pts=True)

    # Проверка 0909 (Section 4.4): VFR с пропущенным внутренним PTS [300, 303, 306, None, 315, 325] -> монотонная шкала и 0.0 расхождения при повторном декодировании
    vfr_pts_seq = [300, 303, 306, None, 315, 325]

    class _VfrMixedStream:
        time_base = Fraction(1, 30)
        average_rate = Fraction(30, 1)
        base_rate = Fraction(30, 1)
        start_time = 300
        duration = 25
        thread_type = None
        codec_context = type("C", (), {"name": "stub"})()

    class _VfrMixedContainer:
        streams = type("S", (), {"video": [_VfrMixedStream()]})()
        duration = 833_333
        def decode(self, **kwargs):
            for p_val in vfr_pts_seq:
                fr = av.VideoFrame.from_ndarray(np.zeros((48, 64, 3), np.uint8), format="bgr24")
                fr.pts = p_val
                yield fr
        def close(self):
            pass

    with patch("app.video_decoder.av.open", side_effect=lambda *a, **kw: _VfrMixedContainer()):
        vb = PyAVVideoDecoder.decode_timeline(vfr_path, strict_pts=False)
        for a_f, b_f in zip(vb.coarse_frames, vb.coarse_frames[1:]):
            assert a_f.timestamp_sec < b_f.timestamp_sec
        fids = [f"F{i:04d}" for i in range(len(vfr_pts_seq))]
        redec = PyAVVideoDecoder.decode_frames_by_ids(
            vfr_path, fids, start_offset_sec=vb.start_offset_sec, nominal_fps=vb.nominal_fps, frame_pts_map=vb.frame_pts_map
        )
        ev_enc = PyAVVideoDecoder.encode_verified_evidence_frames(vb, fids)
        for fid in fids:
            assert redec[fid].timestamp_sec == vb.frame_pts_map[fid]["timestamp_sec"]
        for item in ev_enc:
            assert item["timestamp_sec"] == vb.frame_pts_map[item["frame_id"]]["timestamp_sec"]



# --- AT-09: Отсутствие сценарных эвристик и чтения каталога разметки в производственном коде ---
def test_at09_zero_scenario_heuristics_and_no_evaluation_imports():
    forbidden_tokens = [
        "reference_index",
        "official_labels",
        "team_annotations",
        "open_labels.xlsx",
        '"B1"',
        '"B2"',
        '"C1"',
        '"C2"',
    ]
    for py_path in glob.glob(os.path.join(BASE_DIR, "app", "*.py")):
        src = open(py_path, "r", encoding="utf-8").read()
        for tok in forbidden_tokens:
            assert tok not in src, f"Запрещённый сценарный признак {tok} обнаружен в {py_path}"


# --- AT-10: Перекрытие обзора и проверка взаимодействия человек-объект ---
def test_at10_interaction_requires_person_proximity():
    # Объект далеко от человека (расстояние > 0.22) не должен считаться взаимодействием
    person_box = [0.10, 0.20, 0.30, 0.80]
    far_phone_box = [0.75, 0.75, 0.85, 0.85]
    near_phone_box = [0.22, 0.35, 0.28, 0.45]
    assert _person_object_interaction_distance(person_box, far_phone_box) > 0.40
    assert _person_object_interaction_distance(person_box, near_phone_box) == 0.0


# --- AT-11: Сохранение нескольких независимых событий (в том числе ближе 4 секунд) ---
def test_at11_multiple_independent_events_preserved():
    events = [
        {
            "code": "personal_phone_usage",
            "timecode_sec": 10.0,
            "observed_facts": "Кассир использует личный смартфон за рабочим столом.",
            "evidence_frame_ids": ["F0010"],
        },
        {
            "code": "safe_left_open_during_work",
            "timecode_sec": 11.2,
            "observed_facts": "Дверца сейфа открыта во время работы кассира за столом.",
            "evidence_frame_ids": ["F0012"],
        },
        {
            "code": "personal_phone_usage",
            "timecode_sec": 12.0,
            "observed_facts": "Кассир продолжает держать и использовать личный смартфон за рабочим столом.",
            "evidence_frame_ids": ["F0010", "F0014"],
        },
    ]
    merged = CashDisciplineAnalyzer._merge_same_episode_events(events)
    # Должны сохраниться оба разных типа нарушений (personal_phone_usage и safe_left_open_during_work),
    # а два срабатывания personal_phone_usage одного непрерывного эпизода (с общим кадром F0010) объединиться
    assert len(merged) == 2
    codes = [m["code"] for m in merged]
    assert "personal_phone_usage" in codes
    assert "safe_left_open_during_work" in codes
    phone_ev = [m for m in merged if m["code"] == "personal_phone_usage"][0]
    assert phone_ev["evidence_frame_ids"] == ["F0010", "F0014"]


# --- AT-12: Отклонение ложного кандидата CV моделью VLM ---
def test_at12_vlm_rejects_false_cv_candidate():
    # Проверяем, что валидатор VLM корректно обрабатывает rejected_hypothesis_codes и пустой confirmed_events
    raw_json = json.dumps(
        {
            "scene_description": "Кассир работает за столом в штатном режиме, нарушений нет.",
            "visibility_limitations": [],
            "rejected_hypothesis_codes": ["multiple_clients_in_cash_room"],
            "confirmed_events": [],
            "diagnostic_observations": [],
        },
        ensure_ascii=False,
    )
    parsed = LocalVLMEngine._validate_structured_payload(
        raw_text=raw_json,
        allowed_frame_ids=["F0001", "F0002"],
        duration_sec=20.0,
    )
    assert parsed["confirmed_events"] == []
    assert "multiple_clients_in_cash_room" in parsed["rejected_hypothesis_codes"]


# --- AT-13: Несуществующий кадр, неизвестный/отключённый код или недопустимый таймкод ---
def test_at13_invalid_frame_unknown_code_and_out_of_bounds_timecode_rejected():
    # 1. Несуществующий frame_id F9999
    bad_frame_json = json.dumps(
        {
            "scene_description": "Тест несуществующего кадра",
            "visibility_limitations": [],
            "rejected_hypothesis_codes": [],
            "confirmed_events": [
                {
                    "code": "personal_phone_usage",
                    "timecode_sec": 5.0,
                    "observed_facts": "Кассир держит смартфон в руках за столом.",
                    "evidence_frame_ids": ["F9999"],
                }
            ],
            "diagnostic_observations": [],
        }
    )
    with pytest.raises(ValueError, match="Несуществующий или не показанный"):
        LocalVLMEngine._validate_structured_payload(bad_frame_json, ["F0001", "F0002"], 20.0)

    # 2. Неизвестный код
    bad_code_json = json.dumps(
        {
            "scene_description": "Тест неизвестного кода",
            "visibility_limitations": [],
            "rejected_hypothesis_codes": [],
            "confirmed_events": [
                {
                    "code": "non_existent_code_xyz",
                    "timecode_sec": 5.0,
                    "observed_facts": "Какое-то действие кассира за столом.",
                    "evidence_frame_ids": ["F0001"],
                }
            ],
            "diagnostic_observations": [],
        }
    )
    with pytest.raises(ValueError, match="Неизвестный код"):
        LocalVLMEngine._validate_structured_payload(bad_code_json, ["F0001", "F0002"], 20.0)

    # 3. Таймкод за пределами длительности ролика (запрещено молчаливое обрезание до duration_sec!)
    bad_tc_json = json.dumps(
        {
            "scene_description": "Тест таймкода за пределами видео",
            "visibility_limitations": [],
            "rejected_hypothesis_codes": [],
            "confirmed_events": [
                {
                    "code": "personal_phone_usage",
                    "timecode_sec": 45.0,
                    "observed_facts": "Кассир держит смартфон в руках за столом.",
                    "evidence_frame_ids": ["F0001"],
                }
            ],
            "diagnostic_observations": [],
        }
    )
    with pytest.raises(ValueError, match="выходящий за границы видео"):
        LocalVLMEngine._validate_structured_payload(bad_tc_json, ["F0001", "F0002"], 20.0)

    # 4. Нарушение межполевого инварианта verdict='violation' при пустом violations=[]
    with pytest.raises(ValidationError):
        AnalysisResultModel.model_validate(
            {
                "verdict": "violation",
                "violations": [],
                "meta": {"duration_sec": 15.0},
            }
        )


# --- AT-14: Повреждённый файл и битый JSON приводят к failed, а не к no_violation ---
def test_at14_corrupted_file_and_broken_json():
    corrupt_path = os.path.join(SYNTH_DIR, "corrupted_invalid.mov")
    assert os.path.exists(corrupt_path)
    with pytest.raises(VideoDecodeError):
        PyAVVideoDecoder.decode_timeline(corrupt_path)

    with pytest.raises(ValueError, match="Синтаксическая ошибка JSON"):
        LocalVLMEngine._validate_structured_payload("{broken_json_without_closing_brace", ["F0001"], 10.0)


# --- AT-15: Реальный таймаут без опроса API, отклонение позднего ответа и продолжение очереди ---
def test_at15_real_timeout_without_polling_and_queue_recovery():
    mono_path = os.path.join(SYNTH_DIR, "monotone_valid.mov")
    with tempfile.TemporaryDirectory() as tmp_storage:
        mgr = ProcessJobManager(storage_dir=tmp_storage, default_timeout_sec=0.8, max_queue_size=8)
        try:
            # Отправляем зависающую задачу (sleep 5.0с при таймауте 0.8с) и НЕ опрашиваем её во время работы
            stuck_job = mgr.submit_job(
                video_path=mono_path,
                original_filename="stuck.mov",
                pipeline_mode="cv_only",
                timeout_sec=0.8,
                simulate_sleep_sec=5.0,
            )
            stuck_id = stuck_job["job_id"]
            stuck_token = stuck_job["attempt_token"]

            # Ждём 1.4 секунды без вызовов get_job()
            time.sleep(1.4)

            after_job = mgr.get_job(stuck_id)
            assert after_job is not None
            assert after_job["status"] == "failed"
            assert "timeout" in after_job["internal_state"]

            # Пытаемся подсунуть запоздавший успешный ответ от старого воркера -> должен быть отвергнут!
            accepted_late = mgr.try_apply_worker_message(
                job_id=stuck_id,
                attempt_token=stuck_token,
                ok=True,
                result={"verdict": "no_violation", "violations": [], "meta": {"duration_sec": 6.0}},
            )
            assert accepted_late is False
            assert mgr.get_job(stuck_id)["status"] == "failed"

            # Отправляем следующую нормальную задачу -> новый процесс-работник должен её успешно выполнить
            next_job = mgr.submit_job(
                video_path=mono_path,
                original_filename="normal_next.mov",
                pipeline_mode="cv_only",
                timeout_sec=15.0,
                simulate_sleep_sec=0.0,
            )
            next_id = next_job["job_id"]
            for _ in range(40):
                time.sleep(0.2)
                st = mgr.get_job(next_id)
                if st and st["status"] in ("done", "failed"):
                    break
            final_next = mgr.get_job(next_id)
            assert final_next is not None
            assert final_next["status"] == "done"
            assert final_next["verdict"] == "no_violation"
        finally:
            mgr.shutdown()


# --- AT-16: Перезапуск сервиса переводит незавершённые задачи в failed, а --check не модифицирует jobs_data ---
def test_at16_service_restart_recovers_interrupted_jobs():
    import subprocess

    with tempfile.TemporaryDirectory() as tmp_storage:
        interrupted_id = "interrupted123"
        job_path = os.path.join(tmp_storage, f"{interrupted_id}.json")
        with open(job_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "job_id": interrupted_id,
                    "status": "processing",
                    "internal_state": "running",
                    "accepted_at": time.time() - 10.0,
                },
                f,
            )

        mgr = ProcessJobManager(storage_dir=tmp_storage, default_timeout_sec=60.0)
        try:
            rec = mgr.get_job(interrupted_id)
            assert rec is not None
            assert rec["status"] == "failed"
            assert rec["internal_state"] == "failed_interrupted_on_restart"
        finally:
            mgr.shutdown()

    # Проверка R13: start_service.sh --check не мутирует jobs_data/ побайтово (включая processing-запись и .tmp)
    jobs_dir = os.path.join(BASE_DIR, "jobs_data")
    os.makedirs(jobs_dir, exist_ok=True)
    probe_json = os.path.join(jobs_dir, "_at16_readonly_probe.json")
    probe_tmp = os.path.join(jobs_dir, "_at16_orphan.json.tmp.example")
    try:
        with open(probe_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "job_id": "_at16_readonly_probe",
                    "status": "processing",
                    "internal_state": "running",
                    "deadline_at": time.time() + 300.0,
                },
                f,
            )
        with open(probe_tmp, "w", encoding="utf-8") as f:
            f.write("preserve readonly probe")
        before_bytes = {
            fn: open(os.path.join(jobs_dir, fn), "rb").read()
            for fn in os.listdir(jobs_dir)
            if os.path.isfile(os.path.join(jobs_dir, fn))
        }
        proc = subprocess.run(
            ["/bin/bash", "./start_service.sh", "--check"],
            cwd=BASE_DIR,
            env={"PATH": "/usr/bin:/bin", "HOME": os.path.expanduser("~")},
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, proc.stderr
        after_bytes = {
            fn: open(os.path.join(jobs_dir, fn), "rb").read()
            for fn in os.listdir(jobs_dir)
            if os.path.isfile(os.path.join(jobs_dir, fn))
        }
        assert before_bytes == after_bytes
        assert os.path.exists(probe_tmp)
    finally:
        for p in (probe_json, probe_tmp):
            if os.path.exists(p):
                os.remove(p)


# --- AT-17: Доказательства (1–3 реальных кадра, валидный JPEG base64, совпадение PTS времени) ---
def test_at17_evidence_frames_verified_from_decoded_bundle():
    video_p = os.path.join(OPEN_VIDEOS_DIR, "Х-26.mov")
    bundle = PyAVVideoDecoder.decode_timeline(video_p, target_interval_sec=1.0)
    fids = [bundle.coarse_frames[0].frame_id, bundle.coarse_frames[len(bundle.coarse_frames) // 2].frame_id]
    ev_frames = PyAVVideoDecoder.encode_verified_evidence_frames(bundle, fids)
    assert 1 <= len(ev_frames) <= 3
    for ef in ev_frames:
        assert ef["frame_id"] in bundle.all_decoded_frames
        raw_jpg = base64.b64decode(ef["image_base64"], validate=True)
        with Image.open(io.BytesIO(raw_jpg)) as im:
            assert im.size == (bundle.width, bundle.height)

    # Попытка передать несуществующий frame_id должна выбрасывать ValueError
    with pytest.raises(ValueError, match="Недопустимый frame_id"):
        PyAVVideoDecoder.encode_verified_evidence_frames(bundle, ["F_NONEXISTENT"])


# --- AT-18: Нормативный справочник и блокировка отключённых правил ---
def test_at18_regulatory_catalog_structure_and_disabled_rules():
    enabled = get_enabled_catalog()
    disabled = get_disabled_catalog()
    assert len(enabled) == 9
    assert len(disabled) == 11

    required_fields = {
        "code",
        "type",
        "short_title",
        "regulation_doc",
        "regulation_clause",
        "regulation_url",
        "verified_date",
        "applicability_conditions",
        "mandatory_visual_indicators",
        "inference_limitations",
        "enabled",
        "disabled_reason",
    }
    for code, meta in VIOLATION_CATALOG.items():
        assert required_fields.issubset(set(meta.keys())), f"В правиле {code} не хватает обязательных полей"
        assert meta["regulation_url"].startswith("https://adilet.zan.kz/rus/docs/")
        if not meta["enabled"]:
            assert meta["disabled_reason"] and len(meta["disabled_reason"]) > 15

    # Отключённое правило (включая переведённые по Разделу 4 ревью) не должно проходить validate_rule_applicability для публичного вердикта
    for dis_code in (
        "closing_procedure_lights_off_return",
        "unlocked_workstation_on_exit",
        "food_or_drink_at_workstation",
        "mixed_denominations_packaging",
        "unauthorized_person_in_cashier_zone",
        "unattended_documents_stamps_on_exit",
        "multiple_clients_in_cash_room",
    ):
        with pytest.raises(ValueError, match="отключено"):
            validate_rule_applicability(dis_code, allow_disabled=False)


# --- AT-19: API и пользовательский интерфейс (авторизация, запрет ключа в URL, mode/vlm_mode, internal_state, форматы, XSS) ---
def test_at19_api_and_ui_security(monkeypatch):
    import app.main as main_mod
    client = TestClient(app)

    # 1. Без X-API-Key или со старым публичным ключом из исходников защищённые эндпоинты возвращают 401 (N04)
    assert API_KEY != "forte-datathon26-key"
    assert client.post("/api/v1/analyze").status_code == 401
    assert client.post("/api/v1/analyze-sample", data={"video_id": "Х-26"}).status_code == 401
    assert client.get("/api/v1/jobs/somejob").status_code == 401
    assert client.get("/api/v1/video/somejob").status_code == 401
    assert client.get("/api/v1/dataset").status_code == 401
    assert client.get("/api/v1/dataset", headers={"X-API-Key": "forte-datathon26-key"}).status_code == 401

    # 2. Передача ключа через query-параметр ?api_key=... в видеоплеере ОТКЛОНЯЕТСЯ (401)
    assert client.get(f"/api/v1/video/somejob?api_key={API_KEY}").status_code == 401

    # 3. Публичный каталог и /api/v1/health работают и не раскрывают секретный API_KEY
    cat_res = client.get("/api/v1/catalog")
    assert cat_res.status_code == 200
    assert API_KEY not in cat_res.text
    health_res = client.get("/api/v1/health")
    assert health_res.status_code == 200
    h_json = health_res.json()
    assert h_json["status"] == "ok"
    assert h_json["healthy"] is True
    assert h_json["supervisor_alive"] is True
    assert h_json["vlm_model_cached"] is True
    assert "queue_depth" in h_json and "max_queue_size" in h_json

    # 4. Неподдерживаемый контейнер .avi / .mkv отклоняется с понятным кодом 400
    bad_ext_res = client.post(
        "/api/v1/analyze",
        headers={"X-API-Key": API_KEY},
        files={"file": ("test.avi", b"dummy_avi_bytes", "video/x-msvideo")},
    )
    assert bad_ext_res.status_code == 400
    assert "Неподдерживаемый формат" in bad_ext_res.json()["detail"]

    # 5. Поддержка обоих полей mode и vlm_mode, возврат internal_state и отклонение невалидного режима (400)
    captured_submissions = []
    def _fake_submit_job(**kwargs):
        captured_submissions.append(kwargs)
        return {"job_id": kwargs["job_id"], "status": "processing", "internal_state": "queued"}
    monkeypatch.setattr(main_mod.job_manager, "submit_job", _fake_submit_job)

    for fields, expected_mode in (({"vlm_mode": "deep"}, "deep"), ({"mode": "fast"}, "fast")):
        r_up = client.post(
            "/api/v1/analyze",
            headers={"X-API-Key": API_KEY},
            data=fields,
            files={"file": ("probe.mov", b"probe-container", "video/quicktime")},
        )
        assert r_up.status_code == 200
        assert r_up.json()["internal_state"] == "queued"
        assert captured_submissions[-1]["vlm_mode"] == expected_mode

        r_smp = client.post(
            "/api/v1/analyze-sample",
            headers={"X-API-Key": API_KEY},
            data={"video_id": "Х-26", **fields},
        )
        assert r_smp.status_code == 200
        assert r_smp.json()["internal_state"] == "queued"
        assert captured_submissions[-1]["vlm_mode"] == expected_mode

    bad_mode_res = client.post(
        "/api/v1/analyze",
        headers={"X-API-Key": API_KEY},
        data={"vlm_mode": "invalid_mode"},
        files={"file": ("probe.mov", b"probe-container", "video/quicktime")},
    )
    assert bad_mode_res.status_code == 400

    # 6. Проверка UI (index.html): используется Blob URL (URL.createObjectURL / revokeObjectURL), нет ?api_key= и нет зашитого ключа
    html_text = client.get("/").text
    assert API_KEY not in html_text
    assert "URL.createObjectURL" in html_text
    assert "URL.revokeObjectURL" in html_text
    assert "?api_key=" not in html_text
    assert "escapeHtml" in html_text


# --- AT-20: Локальная работа без интернета и явная ошибка при отсутствии весов ---
def test_at20_offline_model_resolution_and_missing_weights_error(monkeypatch):
    engine = LocalVLMEngine()
    assert engine.is_model_cached() is True

    # Если указать пустую папку весов, движок обязан выбросить явную ошибку ModelNotCachedError (а не лезть в сеть)
    with tempfile.TemporaryDirectory() as empty_dir:
        monkeypatch.setenv("FORTE_VLM_SNAPSHOT_DIR", empty_dir)
        monkeypatch.setattr(
            os.path,
            "expanduser",
            lambda p: empty_dir if "huggingface" in p else os.path.abspath(p),
        )
        fresh_engine = LocalVLMEngine()
        assert fresh_engine.is_model_cached() is False
        with pytest.raises(ModelNotCachedError):
            fresh_engine._load()


# --- AT-21: Согласованность pyproject.toml, requirements.txt, requirements.lock.txt и uv.lock ---
def test_at21_environment_lockfile_consistency():
    import re
    import shutil
    import subprocess
    import sys
    import tomllib
    from pathlib import Path
    from urllib.parse import unquote, urlparse
    from packaging.utils import parse_sdist_filename, parse_wheel_filename

    def _canon(name: str) -> str:
        return re.sub(r"[-_.]+", "-", name).lower()

    req_path = os.path.join(BASE_DIR, "requirements.txt")
    req_lock_path = os.path.join(BASE_DIR, "requirements.lock.txt")
    pyproject_path = os.path.join(BASE_DIR, "pyproject.toml")
    uv_lock_path = os.path.join(BASE_DIR, "uv.lock")

    for p in (req_path, req_lock_path, pyproject_path, uv_lock_path):
        assert os.path.exists(p)

    req_lines = [
        line.strip()
        for line in open(req_path, "r", encoding="utf-8")
        if line.strip() and not line.strip().startswith("#")
    ]
    pyproject_text = open(pyproject_path, "r", encoding="utf-8").read()
    for rline in req_lines:
        assert rline in pyproject_text, f"Зависимость {rline} из requirements.txt должна совпадать с pyproject.toml"

    # R12: Полная синхронизация пинов, wheel/sdist имён файлов и графа зависимостей между requirements.lock.txt и uv.lock
    pip_pins = {}
    for line in open(req_lock_path, "r", encoding="utf-8"):
        line = line.strip()
        if "==" in line and not line.startswith("#"):
            k, v = line.split("==", 1)
            pip_pins[_canon(k.strip())] = v.strip()

    with open(uv_lock_path, "rb") as f:
        uv_data = tomllib.load(f)

    all_uv_pkgs = {_canon(pkg["name"]): pkg["version"] for pkg in uv_data.get("package", [])}
    assert "fortebank-cash-discipline-ai" in all_uv_pkgs

    uv_pins = {}
    for pkg in uv_data.get("package", []):
        cname = _canon(pkg["name"])
        for edge in pkg.get("dependencies", []):
            assert _canon(edge["name"]) in all_uv_pkgs, f"Висячая зависимость {edge} у пакета {pkg['name']}"
        if cname in ("fortebank-cash-discipline-ai", "forte-cash-discipline-ai"):
            continue
        uv_pins[cname] = pkg["version"]
        assert pkg.get("wheels") or pkg.get("sdist"), f"У пакета {pkg['name']} отсутствуют wheels/sdist в uv.lock"
        for w in pkg.get("wheels", []):
            fname = unquote(Path(urlparse(w["url"]).path).name)
            w_name, w_ver, *_ = parse_wheel_filename(fname)
            assert _canon(w_name) == cname and str(w_ver) == pkg["version"], (
                f"Несовпадение версии wheel {fname} с пакетом {pkg['name']}=={pkg['version']}"
            )
        if pkg.get("sdist"):
            fname = unquote(Path(urlparse(pkg["sdist"]["url"]).path).name)
            s_name, s_ver = parse_sdist_filename(fname)
            assert _canon(s_name) == cname and str(s_ver) == pkg["version"], (
                f"Несовпадение версии sdist {fname} с пакетом {pkg['name']}=={pkg['version']}"
            )

    assert pip_pins == uv_pins
    assert "aiohttp" in pip_pins

    uv_bin = shutil.which("uv") or "/Users/aliaskaravgust/.local/bin/uv"
    if os.path.exists(uv_bin):
        uv_proc = subprocess.run(
            [
                uv_bin,
                "sync",
                "--frozen",
                "--dry-run",
                "--offline",
                "--no-cache",
                "--no-python-downloads",
                "--python",
                sys.executable,
            ],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=25,
        )
        assert uv_proc.returncode == 0, uv_proc.stderr

    # R13: start_service.sh --check работает при PATH=/usr/bin:/bin и НЕ модифицирует jobs_data/
    proc = subprocess.run(
        ["/bin/bash", "./start_service.sh", "--check"],
        cwd=BASE_DIR,
        env={"PATH": "/usr/bin:/bin", "HOME": os.path.expanduser("~")},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr


# --- AT-22: Полный 4-частный инженерный отчёт и артефакты оценки ---
def test_at22_full_4part_report_and_evaluation_artifacts():
    import hashlib
    import statistics
    from evaluation.run_stage5_evaluation import compute_metrics_summary

    report_path = os.path.join(BASE_DIR, "REPORT.md")
    open42_path = os.path.join(BASE_DIR, "evaluation", "results", "stage5_open42_ablation.json")
    aug_path = os.path.join(BASE_DIR, "evaluation", "results", "stage5_augmentations_eval.json")
    meva_path = os.path.join(BASE_DIR, "evaluation", "results", "stage6_meva_evaluation.json")

    for p in (report_path, open42_path, aug_path, meva_path):
        assert os.path.exists(p), f"Обязательный артефакт отчёта отсутствует: {p}"

    report_text = open(report_path, "r", encoding="utf-8").read()
    report_lines = set(report_text.splitlines())
    for section_title in ("Часть 1.", "Часть 2.", "Часть 3.", "Часть 4."):
        assert section_title in report_text, f"В REPORT.md отсутствует раздел '{section_title}'"
    assert "/Users/aliaskaravgust/" not in report_text

    with open(os.path.join(BASE_DIR, "evaluation", "official_labels.json"), "r", encoding="utf-8") as f:
        official = json.load(f)["items"]
    with open(os.path.join(BASE_DIR, "evaluation", "team_annotations.json"), "r", encoding="utf-8") as f:
        team = json.load(f)["items"]

    def _bin_row(label: str, b: dict, include_end_to_end: bool = True) -> str:
        tot = b["total_inputs"]
        succ = b["succeeded_count"]
        fail = b["failed_count"]
        tp, fp, tn, fn = b["TP"], b["FP"], b["TN"], b["FN"]
        prec = f"`{b['precision']:.4f}` (`{tp}/{tp+fp}`)" if (tp + fp) > 0 else "`0.0000` (`0/0`)"
        rec = f"`{b['recall']:.4f}` (`{tp}/{tp+fn}`)"
        f1 = f"`{b['f1']:.4f}`"
        acc = f"**`{b['accuracy_on_succeeded']*100:.2f}%` (`{tp+tn}/{succ}`)**"
        if include_end_to_end:
            e2e = f"**`{b['end_to_end_correct_rate_all_inputs']*100:.2f}%` (`{tp+tn}/{tot}`)**"
            return f"| {label} | `{tot} / {succ} / {fail}` | {tp} | {fp} | {tn} | {fn} | {prec} | {rec} | {f1} | {acc} | {e2e} |"
        return f"| {label} | `{tot} / {succ} / {fail}` | {tp} | {fp} | {tn} | {fn} | {prec} | {rec} | {f1} | {acc} |"

    discrepancies = []

    with open(open42_path, "r", encoding="utf-8") as f:
        open42 = json.load(f)
    cfg_labels = {
        "cv_only": "**`cv_only`**",
        "vlm_only": "**`vlm_only`**",
        "hybrid": "**`hybrid` (CV + VLM)**",
    }
    for cfg, row_label in cfg_labels.items():
        assert cfg in open42["configurations"]
        recs = open42["configurations"][cfg]["per_video_records"]
        assert len(recs) == 42
        recomp = compute_metrics_summary(recs, official, team)
        for k, v in recomp.items():
            if k != "operational_telemetry":
                if open42["configurations"][cfg]["summary"][k] != v:
                    discrepancies.append(f"open42/{cfg}/{k} mismatch")
        s = open42["configurations"][cfg]["summary"]
        expected_rows = (
            ("2.1.A_all42", _bin_row(row_label, s["all_42_official_labels"], True)),
            ("2.1.B_uniq40", _bin_row(row_label, s["unique_40_groups_official_labels"], False)),
            ("2.1.C_nbk42", _bin_row(f"{row_label} | Все 42 файла (`enabled=True` GT)", s["all_42_public_nbk_enabled_rules"], False)),
            ("2.1.C_nbk40", _bin_row(f"{row_label} | 40 уникальных групп (`enabled=True` GT)", s["unique_40_public_nbk_enabled_rules"], False)),
        )
        for table_tag, exp_row in expected_rows:
            if exp_row not in report_lines:
                discrepancies.append(f"REPORT.md missing exact row for {cfg}/{table_tag}: {exp_row}")

    # N03 (ревью 1017): Полная проверка таблицы 2.2 (сопоставление категорий и MAE времени по cv_only, vlm_only, hybrid)
    s_cv = open42["configurations"]["cv_only"]["summary"]
    s_vlm = open42["configurations"]["vlm_only"]["summary"]
    s_hyb = open42["configurations"]["hybrid"]["summary"]
    c_cv, c_vlm, c_hyb = (
        s_cv["category_and_event_matching"],
        s_vlm["category_and_event_matching"],
        s_hyb["category_and_event_matching"],
    )
    t_cv, t_vlm, t_hyb = (
        s_cv["temporal_localization_on_positives"],
        s_vlm["temporal_localization_on_positives"],
        s_hyb["temporal_localization_on_positives"],
    )

    def _fmt_w2_matched(t_block: dict) -> str:
        ev_cnt = t_block["evaluated_matched_events_count"]
        if ev_cnt > 0:
            return f"`{t_block['within_2_0s_count']} / {ev_cnt}` (`{t_block['within_2_0s_rate_on_matched']*100:.2f}%`)"
        return "не применимо (`0/0`)"

    def _fmt_w5_matched(t_block: dict) -> str:
        ev_cnt = t_block["evaluated_matched_events_count"]
        if ev_cnt > 0:
            return f"`{t_block['within_5_0s_count']} / {ev_cnt}`"
        return "не применимо (`0/0`)"

    def _fmt_sec_or_dash(val: object) -> str:
        return f"`{val} с`" if val is not None else "—"

    sec22_expected_rows = (
        (
            "2.2_total_annotated",
            f"| Всего размеченных событий в `team_annotations.json` (по всем 42 входам) | `{c_cv['total_expected_all_events']}` | `{c_vlm['total_expected_all_events']}` | `{c_hyb['total_expected_all_events']}` |",
        ),
        (
            "2.2_enabled_denom",
            f"| Из них активных публичных событий НБРК (`enabled=True`, фиксированный знаменатель) | `{c_cv['total_expected_enabled_public_events']}` | `{c_vlm['total_expected_enabled_public_events']}` | `{c_hyb['total_expected_enabled_public_events']}` |",
        ),
        (
            "2.2_matched_category",
            f"| Сопоставлено событий с точным совпадением кода категории (`matched_exact_category_events`) | `{c_cv['matched_exact_category_events']} / {c_cv['total_expected_enabled_public_events']}` (`{c_cv['exact_category_recall_on_enabled']*100:.2f}%`) | `{c_vlm['matched_exact_category_events']} / {c_vlm['total_expected_enabled_public_events']}` (`{c_vlm['exact_category_recall_on_enabled']*100:.2f}%`) | `{c_hyb['matched_exact_category_events']} / {c_hyb['total_expected_enabled_public_events']}` (`{c_hyb['exact_category_recall_on_enabled']*100:.2f}%`) |",
        ),
        (
            "2.2_missed_events",
            f"| Пропущено активных событий (`missed_enabled_events`) | `{c_cv['missed_enabled_events']}` | `{c_vlm['missed_enabled_events']}` | `{c_hyb['missed_enabled_events']}` |",
        ),
        (
            "2.2_extra_events",
            f"| Лишних предсказанных событий (`extra_predicted_events`, включая смену категории) | `{c_cv['extra_predicted_events']}` | `{c_vlm['extra_predicted_events']}` | `{c_hyb['extra_predicted_events']}` |",
        ),
        (
            "2.2_evaluated_matched",
            f"| Число сопоставленных одноимённых событий для оценки времени | `{t_cv['evaluated_matched_events_count']}` | `{t_vlm['evaluated_matched_events_count']}` | `{t_hyb['evaluated_matched_events_count']}` |",
        ),
        (
            "2.2_within_2s_matched",
            f"| Событий с ошибкой `\\|t_pred - t_gt\\| <= 2.0s` (от сопоставленных) | {_fmt_w2_matched(t_cv)} | {_fmt_w2_matched(t_vlm)} | {_fmt_w2_matched(t_hyb)} |",
        ),
        (
            "2.2_within_2s_all_enabled",
            f"| Событий с ошибкой `\\|t_pred - t_gt\\| <= 2.0s` (от всех `{c_hyb['total_expected_enabled_public_events']}` активных GT-событий) | `{t_cv['within_2_0s_count']} / {c_cv['total_expected_enabled_public_events']}` (`{t_cv['within_2_0s_rate_on_all_enabled_expected']*100:.2f}%`) | `{t_vlm['within_2_0s_count']} / {c_vlm['total_expected_enabled_public_events']}` (`{t_vlm['within_2_0s_rate_on_all_enabled_expected']*100:.2f}%`) | `{t_hyb['within_2_0s_count']} / {c_hyb['total_expected_enabled_public_events']}` (`{t_hyb['within_2_0s_rate_on_all_enabled_expected']*100:.2f}%`) |",
        ),
        (
            "2.2_within_5s_matched",
            f"| Событий с ошибкой `\\|t_pred - t_gt\\| <= 5.0s` (от сопоставленных) | {_fmt_w5_matched(t_cv)} | {_fmt_w5_matched(t_vlm)} | {_fmt_w5_matched(t_hyb)} |",
        ),
        (
            "2.2_mae",
            f"| Средняя абсолютная ошибка времени `MAE` (с) | {_fmt_sec_or_dash(t_cv['mean_abs_error_sec'])} | {_fmt_sec_or_dash(t_vlm['mean_abs_error_sec'])} | {_fmt_sec_or_dash(t_hyb['mean_abs_error_sec'])} |",
        ),
        (
            "2.2_median_ae",
            f"| Медианная абсолютная ошибка времени `Median AE` (с) | {_fmt_sec_or_dash(t_cv['median_abs_error_sec'])} | {_fmt_sec_or_dash(t_vlm['median_abs_error_sec'])} | {_fmt_sec_or_dash(t_hyb['median_abs_error_sec'])} |",
        ),
        (
            "2.2_max_ae",
            f"| Максимальная абсолютная ошибка времени `Max AE` (с) | {_fmt_sec_or_dash(t_cv['max_abs_error_sec'])} | {_fmt_sec_or_dash(t_vlm['max_abs_error_sec'])} | {_fmt_sec_or_dash(t_hyb['max_abs_error_sec'])} |",
        ),
        (
            "2.2_abs_errors_list",
            f"| Список ошибок времени по сопоставленным одноимённым событиям (с) | `{t_cv['abs_errors_sec_list']}` | `{t_vlm['abs_errors_sec_list']}` | `{t_hyb['abs_errors_sec_list']}` |",
        ),
    )
    for sec22_tag, exp_sec22_row in sec22_expected_rows:
        if exp_sec22_row not in report_lines:
            discrepancies.append(f"REPORT.md missing exact Section 2.2 row ({sec22_tag}): {exp_sec22_row}")

    with open(aug_path, "r", encoding="utf-8") as f:
        aug = json.load(f)
    assert aug["manifest_total_files"] == 210
    assert aug["cv_only_all_210"]["total_evaluated_files"] == 210
    assert aug["hybrid_pilot_50"]["total_evaluated_files"] == 50

    aug_labels = {
        "scale_2x": "**`scale_2x`** (`iw*2:ih*2`, bicubic)",
        "scale_075x": "**`scale_075x`** (`iw*0.75:ih*0.75`)",
        "prefix_5s": "**`prefix_5s`** (`+5.0s` нейтральное начало)",
        "luma_contrast": "**`luma_contrast`** (`brightness=+0.05, contrast=1.08`)",
        "h264_compression": "**`h264_compression`** (`CRF 28`)",
    }

    for sec_key in ("cv_only_all_210", "hybrid_pilot_50"):
        sec = aug[sec_key]
        records = sec["records"]
        if len(records) != sec["total_evaluated_files"]:
            discrepancies.append(f"aug/{sec_key} records count {len(records)} != {sec['total_evaluated_files']}")
        for aug_type, b_sum in sec["by_augmentation_type"].items():
            tf_recs = [r for r in records if r["augmentation_type"] == aug_type]
            strict_recs = [r for r in tf_recs if r["semantic_validity"] == "valid"]
            flagged_recs = [r for r in tf_recs if r["semantic_validity"] != "valid"]
            v_ok = sum(1 for r in strict_recs if r["verdict_consistent_with_source"])
            c_ok = sum(1 for r in strict_recs if r["category_consistent_with_source"])
            src_ev = sum(len(r["source_violations"]) for r in strict_recs)
            shifts = [err for r in strict_recs for err in r["time_shift_errors_vs_source_sec"]]
            w2_cnt = sum(1 for err in shifts if err <= 2.0)
            miss_cnt = sum(r["missed_source_events_count"] for r in strict_recs)
            extra_cnt = sum(r["extra_predicted_events_count"] for r in strict_recs)

            expected_aug_summary = {
                "total_files": len(tf_recs),
                "unique_origin_groups": len({r["origin_group_id"] for r in tf_recs}),
                "strict_semantically_valid_count": len(strict_recs),
                "strict_valid_succeeded_count": sum(1 for r in strict_recs if r["status"] == "done"),
                "strict_valid_failed_count": sum(1 for r in strict_recs if r["status"] != "done"),
                "flagged_degraded_or_illumination_count": len(flagged_recs),
                "verdict_consistency_rate": round(v_ok / max(1, len(strict_recs)), 4),
                "verdict_consistency_fraction": f"{v_ok}/{len(strict_recs)}",
                "category_consistency_rate": round(c_ok / max(1, len(strict_recs)), 4),
                "category_consistency_fraction": f"{c_ok}/{len(strict_recs)}",
                "source_events_total": src_ev,
                "temporal_shift_evaluated_events": len(shifts),
                "missed_source_events_count": miss_cnt,
                "extra_predicted_events_count": extra_cnt,
                "temporal_shift_within_2_0s_count": w2_cnt,
                "temporal_shift_within_2_0s_rate": round(w2_cnt / len(shifts), 4) if shifts else None,
                "temporal_shift_within_2_0s_rate_on_all_source_events": round(w2_cnt / src_ev, 4) if src_ev > 0 else None,
                "temporal_shift_mean_abs_error_sec": round(statistics.mean(shifts), 2) if shifts else None,
            }
            for k, exp_v in expected_aug_summary.items():
                if b_sum.get(k) != exp_v:
                    discrepancies.append(f"aug/{sec_key}/{aug_type}/{k} {b_sum.get(k)} != {exp_v}")

            w2_matched = (
                f"`{expected_aug_summary['temporal_shift_within_2_0s_count']}/{expected_aug_summary['temporal_shift_evaluated_events']}` (`{expected_aug_summary['temporal_shift_within_2_0s_rate']*100:.2f}%`)"
                if expected_aug_summary["temporal_shift_evaluated_events"] > 0
                else "не применимо (`0/0`)"
            )
            w2_all_src = (
                f"`{expected_aug_summary['temporal_shift_within_2_0s_count']}/{expected_aug_summary['source_events_total']}` (`{expected_aug_summary['temporal_shift_within_2_0s_rate_on_all_source_events']*100:.2f}%`)"
                if expected_aug_summary["source_events_total"] > 0
                else "не применимо (`0/0`)"
            )
            mae_str = (
                f"`{expected_aug_summary['temporal_shift_mean_abs_error_sec']:.2f} с`"
                if expected_aug_summary["temporal_shift_mean_abs_error_sec"] is not None
                else "—"
            )
            exp_aug_row = (
                f"| {aug_labels[aug_type]} | `{expected_aug_summary['total_files']} / {expected_aug_summary['unique_origin_groups']}` | "
                f"`{expected_aug_summary['strict_semantically_valid_count']}` (`done={expected_aug_summary['strict_valid_succeeded_count']}, failed={expected_aug_summary['strict_valid_failed_count']}`) | "
                f"`{expected_aug_summary['flagged_degraded_or_illumination_count']}` | "
                f"`{expected_aug_summary['verdict_consistency_fraction']}` (`{expected_aug_summary['verdict_consistency_rate']*100:.2f}%`) | "
                f"`{expected_aug_summary['category_consistency_fraction']}` (`{expected_aug_summary['category_consistency_rate']*100:.2f}%`) | "
                f"`{expected_aug_summary['source_events_total']} / {expected_aug_summary['temporal_shift_evaluated_events']} / {expected_aug_summary['missed_source_events_count']} / {expected_aug_summary['extra_predicted_events_count']}` | "
                f"{w2_matched} | {w2_all_src} | {mae_str} |"
            )
            if exp_aug_row not in report_lines:
                discrepancies.append(f"REPORT.md missing exact aug row {sec_key}/{aug_type}")

    with open(meva_path, "r", encoding="utf-8") as f:
        meva = json.load(f)
    assert meva["counts_summary"]["total_episodes"] == 12
    eps = meva["per_episode_results"]
    assert len(eps) == 12
    bars = meva["base_action_recognition_summary"]
    tot_gt = sum(len(ep["official_gt_activities"]) for ep in eps)
    matched_acts = [m for ep in eps for m in ep["matched_activities"]]
    mid_errs = [m["midpoint_error_sec"] for m in matched_acts]
    w2_meva = sum(1 for e in mid_errs if e <= 2.0)
    w3_meva = sum(1 for e in mid_errs if e <= 3.0)

    expected_bars = {
        "total_official_gt_activities": tot_gt,
        "matched_activities_count": len(matched_acts),
        "base_action_recall": round(len(matched_acts) / max(1, tot_gt), 4),
        "base_action_recall_fraction": f"{len(matched_acts)}/{tot_gt}",
        "temporal_localization_evaluated_count": len(mid_errs),
        "midpoint_error_within_2_0s_count": w2_meva,
        "midpoint_error_within_2_0s_rate": round(w2_meva / max(1, len(mid_errs)), 4) if mid_errs else None,
        "midpoint_error_within_3_0s_count": w3_meva,
        "mean_midpoint_error_sec": round(statistics.mean(mid_errs), 2) if mid_errs else None,
    }
    for k, exp_v in expected_bars.items():
        if bars.get(k) != exp_v:
            discrepancies.append(f"meva/bars/{k} {bars.get(k)} != {exp_v}")
    if "midpoint_error_mean_abs_sec" in bars and bars["midpoint_error_mean_abs_sec"] != expected_bars["mean_midpoint_error_sec"]:
        discrepancies.append(f"meva/bars/midpoint_error_mean_abs_sec {bars['midpoint_error_mean_abs_sec']} != {expected_bars['mean_midpoint_error_sec']}")

    meva_b = meva["unfamiliar_scene_banking_diagnostic_signal"]
    total_eps = len(eps)
    b_failed = sum(1 for ep in eps if ep["unfamiliar_scene_banking_api_behavior"].get("banking_status", "done") != "done")
    b_eval = total_eps - b_failed
    b_trig = sum(
        1
        for ep in eps
        if ep["unfamiliar_scene_banking_api_behavior"].get("banking_status", "done") == "done"
        and ep["unfamiliar_scene_banking_api_behavior"]["banking_verdict"] == "violation"
    )
    b_no_viol = sum(
        1
        for ep in eps
        if ep["unfamiliar_scene_banking_api_behavior"].get("banking_status", "done") == "done"
        and ep["unfamiliar_scene_banking_api_behavior"]["banking_verdict"] == "no_violation"
    )
    expected_meva_b = {
        "total_episodes": total_eps,
        "evaluated_episodes_count": b_eval,
        "banking_failed_count": b_failed,
        "banking_no_violation_count": b_no_viol,
        "banking_violation_trigger_count": b_trig,
        "banking_trigger_diagnostic_rate": round(b_trig / b_eval, 4) if b_eval > 0 else None,
        "false_banking_violation_rate_on_evaluated": round(b_trig / b_eval, 4) if b_eval > 0 else None,
        "banking_no_violation_rate_on_all_inputs": round(b_no_viol / max(1, total_eps), 4),
    }
    for k, exp_v in expected_meva_b.items():
        if meva_b.get(k) != exp_v:
            discrepancies.append(f"meva/banking/{k} {meva_b.get(k)} != {exp_v}")

    expected_meva_recall_line = (
        f"сопоставлено **`{expected_bars['base_action_recall_fraction']}` (`{expected_bars['base_action_recall']*100:.2f}%`)** официальных действий MEVA "
        f"(при строгом глобальном взаимно однозначном сопоставлении); средняя ошибка центра интервала по сопоставленным действиям "
        f"`MAE = {expected_bars['mean_midpoint_error_sec']} с` (`{expected_bars['midpoint_error_within_2_0s_count']}/{expected_bars['temporal_localization_evaluated_count']}` сопоставленных действий или "
        f"**`{expected_bars['midpoint_error_within_2_0s_count']}/{expected_bars['total_official_gt_activities']}` всех ожидаемых активностей** в пределах `<= 2.0 с`, "
        f"`{expected_bars['midpoint_error_within_3_0s_count']}/{expected_bars['temporal_localization_evaluated_count']}` в пределах `<= 3.0 с`)."
    )
    if expected_meva_recall_line not in report_text:
        discrepancies.append("REPORT.md missing or corrupted MEVA recall & MAE summary line in Section 4.2")

    meva_no_viol_pct = f"{expected_meva_b['banking_no_violation_count'] / max(1, expected_meva_b['total_episodes']) * 100.0:.1f}%"
    expected_meva_banking_line = (
        f"Из `{expected_meva_b['total_episodes']}` эпизодов успешно оценено `{expected_meva_b['evaluated_episodes_count']}` (`banking_failed_count = {expected_meva_b['banking_failed_count']}`). "
        f"На **`{expected_meva_b['banking_no_violation_count']} / {expected_meva_b['total_episodes']}` (`{meva_no_viol_pct}`)** небанковских эпизодов MEVA"
    )
    if expected_meva_banking_line not in report_text:
        discrepancies.append("REPORT.md missing or corrupted MEVA banking summary line in Section 4.2")

    for doc_name, doc_obj in (("open42", open42), ("aug", aug), ("meva", meva)):
        for rel_code, expected_sha in doc_obj["provenance"]["code_sha256"].items():
            actual_sha = hashlib.sha256(open(os.path.join(BASE_DIR, rel_code), "rb").read()).hexdigest()
            if actual_sha != expected_sha:
                discrepancies.append(f"{doc_name} provenance hash mismatch for {rel_code}")

    assert not discrepancies, " | ".join(discrepancies)


# --- AT-23: Поведенческие контрпримеры ревью (F01..F05, F13, Section 6) ---
def test_at23_pipeline_cpu_counterexamples_and_queue_sweeper():
    import numpy as np
    from unittest.mock import patch
    from app.schemas import CandidateEvent
    from app.video_decoder import DecodedFrame, VideoTimelineBundle

    frames = [
        DecodedFrame(f"F{i:04d}", i, i, float(i), float(i), 16, 16, np.zeros((16, 16, 3), dtype=np.uint8))
        for i in range(120)
    ]
    bundle = VideoTimelineBundle(
        "stub.mov",
        "stub",
        16,
        16,
        1.0,
        "1/1",
        1.0,
        0.0,
        "zero_based_pts",
        120.0,
        False,
        1.0,
        "coarse interval 1.00 sec",
        frames,
        {f.frame_id: f for f in frames},
    )

    # 1. C06: ограниченный шаг между кадрами обзорной раскадровки VLM (<= 4.0s)
    selected = CashDisciplineAnalyzer._select_coarse_storyboard_frames(bundle, 9)
    max_gap = max(b.timestamp_sec - a.timestamp_sec for a, b in zip(selected, selected[1:]))
    assert max_gap <= 4.0

    # 2. C07: отклонение null-полей в JSON-ответе VLM
    null_payload = json.dumps(
        {
            "scene_description": None,
            "visibility_limitations": None,
            "rejected_hypothesis_codes": None,
            "confirmed_events": None,
            "diagnostic_observations": None,
        }
    )
    with pytest.raises(ValueError, match="scene_description"):
        LocalVLMEngine._validate_structured_payload(null_payload, ["F0000"], 20.0)

    # 3. C08 & C09 & C01: отклонение выходящего за границы времени, несвязанного кадра и противоречащих фактов
    def make_out(ev_list):
        return json.dumps(
            {
                "is_banking_cash_room_scene": True,
                "scene_description": "Тестовая сцена",
                "visibility_limitations": [],
                "rejected_hypothesis_codes": [],
                "confirmed_events": ev_list,
                "diagnostic_observations": [],
            }
        )

    with pytest.raises(ValueError, match="выходящий за границы видео"):
        LocalVLMEngine._validate_structured_payload(
            make_out([{"code": "personal_phone_usage", "timecode_sec": 20.2, "observed_facts": "Сотрудник использует телефон.", "evidence_frame_ids": ["F0000"]}]),
            ["F0000"],
            20.0,
        )

    with pytest.raises(ValueError, match="не соответствует временным меткам"):
        LocalVLMEngine._validate_structured_payload(
            make_out([{"code": "personal_phone_usage", "timecode_sec": 80.0, "observed_facts": "Сотрудник использует телефон.", "evidence_frame_ids": ["F0000"]}]),
            ["F0000"],
            120.0,
        )

    encoded_f0 = PyAVVideoDecoder.encode_verified_evidence_frames(bundle, ["F0000"])
    with pytest.raises(ValueError, match="противоречит|ограничение|не подтверждает"):
        format_violation_entry(
            "personal_phone_usage",
            0.0,
            encoded_f0,
            "Клиент держит телефон, сотрудник не использует его.",
        )

    # 4. C10: сохранение выбранного моделью доказательного кадра в _ensure_multiphase_evidence_ids
    phases = CashDisciplineAnalyzer._ensure_multiphase_evidence_ids(
        "unattended_cash_on_desk_on_exit", ["F0011"], frames[5:15], 10.0
    )
    assert "F0011" in phases

    # 5. C11: сохранение независимых одноимённых эпизодов разных сотрудников
    merged = CashDisciplineAnalyzer._merge_same_episode_events(
        [
            {"code": "personal_phone_usage", "timecode_sec": 10.0, "evidence_frame_ids": ["F0010"], "observed_facts": "Первый сотрудник использует телефон."},
            {"code": "personal_phone_usage", "timecode_sec": 12.0, "evidence_frame_ids": ["F0012"], "observed_facts": "Другой сотрудник отдельно использует телефон у второго стола."},
        ]
    )
    assert len(merged) == 2

    # 6. C05: отсутствие позиционного подавления человека в левом верхнем углу кадра
    main_p = {"conf": 0.95, "nbbox": [0.4, 0.3, 0.7, 0.9]}
    left_p = {"conf": 0.90, "nbbox": [0.1, 0.1, 0.2, 0.3]}
    assert len(CVSceneAnalyzer._filter_normalized_persons([main_p, left_p])) == 2

    # 7. F13: истечение таймаута ожидающей в очереди задачи обновляет статус на диске во время работы первой задачи
    with tempfile.TemporaryDirectory() as storage:
        mgr = ProcessJobManager(storage_dir=storage)
        try:
            mgr.submit_job("unused.mov", "blocking.mov", pipeline_mode="cv_only", timeout_sec=2.5, simulate_sleep_sec=10)
            second = mgr.submit_job("unused.mov", "queued.mov", pipeline_mode="cv_only", timeout_sec=0.25, simulate_sleep_sec=10)
            time.sleep(0.8)
            with open(os.path.join(storage, f"{second['job_id']}.json"), "r", encoding="utf-8") as f:
                saved = json.load(f)
            assert saved["status"] == "failed"
            assert saved["internal_state"] == "failed_queue_timeout"
        finally:
            mgr.shutdown()


# --- AT-24: Проверка фиксированного знаменателя для failed и переносимость путей манифестов (F08..F10, F15) ---
def test_at24_evaluation_metrics_and_portable_manifests():
    from evaluation.run_stage5_evaluation import compute_metrics_summary

    with open(os.path.join(BASE_DIR, "evaluation", "official_labels.json"), "r", encoding="utf-8") as f:
        official = json.load(f)["items"]
    with open(os.path.join(BASE_DIR, "evaluation", "team_annotations.json"), "r", encoding="utf-8") as f:
        team = json.load(f)["items"]

    # Проверяем, что упавшая задача ('failed') на положительном ролике Х-26 НЕ исключается из знаменателя событий
    failed_probe = compute_metrics_summary(
        [{"stem": "Х-26", "status": "failed", "verdict": None, "violations": [], "processing_time_sec": 0.0}],
        official,
        team,
    )
    assert failed_probe["category_and_event_matching"]["total_expected_enabled_public_events"] == 1
    assert failed_probe["category_and_event_matching"]["missed_enabled_events"] == 1
    assert failed_probe["all_42_official_labels"]["end_to_end_correct_rate_all_inputs"] == 0.0

    # Проверяем, что все пути в манифестах аугментаций и MEVA относительные и переносимые
    with open(os.path.join(AUG_DIR, "manifest.json"), "r", encoding="utf-8") as f:
        aug_m = json.load(f)
    for e in aug_m["entries"]:
        assert not os.path.isabs(e["augmented_path"])
        assert os.path.exists(os.path.join(BASE_DIR, e["augmented_path"]))

    with open(os.path.join(BASE_DIR, "evaluation", "meva_sample", "manifest.json"), "r", encoding="utf-8") as f:
        meva_m = json.load(f)
    for ep in meva_m["episodes"]:
        assert not os.path.isabs(ep["clip_path"])
        assert not os.path.isabs(ep["local_annotation_path"])
        assert os.path.exists(os.path.join(BASE_DIR, ep["local_annotation_path"]))


# --- AT-25: Сквозной регрессионный тест повторных ревью (R01..R15 + 0909 Sections 4.1..4.5) ---
def test_at25_review_0550_all_regression_probes(monkeypatch):
    import numpy as np
    from unittest.mock import patch
    from app.cv_detector import CVDetectorError
    from app.regulations import validate_observation_consistency
    from app.schemas import CandidateEvent
    from app.video_decoder import DecodedFrame, VideoTimelineBundle
    from evaluation.run_stage5_evaluation import compute_metrics_summary

    # 1. R01: Инвариантность нормализованных координат боксов при разных исходных разрешениях (640x360 vs 1920x1080 vs 3840x2160)
    class _Arr(list):
        def tolist(self):
            return list(self)

    class _Box:
        def __init__(self, c, cf, xy):
            self.cls = np.array([c])
            self.conf = np.array([cf])
            self.xyxy = [_Arr(xy)]

    class _Res:
        boxes = [_Box(0, 0.95, [100.0, 40.0, 260.0, 320.0]), _Box(67, 0.90, [520.0, 260.0, 560.0, 310.0])]

    class _FakeYOLO:
        def predict(self, imgs, *a, **kw):
            return [_Res() for _ in imgs]

    cv = CVSceneAnalyzer()
    cv.yolo_model = _FakeYOLO()
    counts_by_res = []
    for orig_w, orig_h in ((640, 360), (1920, 1080), (3840, 2160)):
        frames = [
            DecodedFrame(f"F{i:04d}", i, i, float(i), float(i), orig_w, orig_h, np.zeros((360, 640, 3), dtype=np.uint8))
            for i in range(6)
        ]
        b = VideoTimelineBundle(
            "s.mov", "s", orig_w, orig_h, 1.0, "1/1", 1.0, 0.0, "zero_based_pts", 6.0, False, 1.0, "coarse", frames, {f.frame_id: f for f in frames}
        )
        out = cv.analyze_timeline(b)
        counts_by_res.append((out["max_persons"], out["detected_interacting_objects"], len(out["candidates"])))
    assert counts_by_res[0] == counts_by_res[1] == counts_by_res[2] == (1, [], 0)

    # 2. R03: Явный CVDetectorError при отсутствии весов и при сбое predict()
    bad_cv = CVSceneAnalyzer(yolo_weights="/nonexistent/yolo11s.pt")
    with pytest.raises(CVDetectorError):
        bad_cv._ensure_yolo()

    # 3. R02 & R05 & 0909 (Sections 4.1 & 4.2): Поусловный валидатор с привязкой актор-действие-объект и запретом обхода через CV-телеметрию
    for bad_code, bad_obs in (
        ("personal_phone_usage", "На рабочем столе лежит прямоугольный предмет."),
        ("personal_phone_usage", "Посетитель смотрит на смартфон; кассир работает с документами."),
        ("unattended_cash_on_desk_on_exit", "Сотрудник встал и вышел; поверхность стола не видна."),
        ("safe_left_open_during_work", "Открытый сейф виден у стола; содержимое и текущая операция неразличимы."),
        ("personal_phone_usage", "Смартфон лежит на столе, кассир считает банкноты."),
        ("personal_phone_usage", "Клиент разговаривает по смартфону; кассир пересчитывает наличные."),
        ("personal_phone_usage", "Кассир смартфоном не пользуется, аппарат лежит на столе."),
        ("safe_left_open_during_work", "Сейф открыт, кассир работает за столом; содержимое скрыто дверцей."),
        ("safe_left_open_on_exit", "Сейф виден у стены, кассир вышел из помещения."),
        ("counting_in_client_absence", "Кассир считает деньги, клиент стоит рядом и наблюдает."),
        ("unattended_previous_client_cash", "Клиент подошел к столу; кассир оформляет документы."),
        ("collection_bag_accepted_without_check", "Кассир принимает сумку; пломбу осмотрели, документы сверили."),
        # Контрпримеры ревью 0909 (Sections 4.1 & 4.2):
        ("personal_phone_usage", "Сотрудник в рабочей зоне взаимодействует с мобильным телефоном (детекция смартфона вблизи рук на 5 кадрах окна)."),
        ("counting_in_client_absence", "Клиент считает деньги за столом, пока кассир отошел от рабочего места в сторону двери."),
        ("personal_phone_usage", "Клиент держит телефон у уха, а кассир смотрит на экран монитора."),
        ("unattended_collection_bag_on_desk", "На столе лежит сумка, а инкассатор проверяет пломбу перед передачей."),
        ("safe_left_open_on_exit", "На рабочем столе кассира лежит пачка денег; кассир отошел, но двери не закрыл."),
        # Контрпримеры ревью 1017 (N02):
        ("personal_phone_usage", "Кассир не держит смартфон в руках."),
        ("unattended_cash_on_desk_on_exit", "Банкноты лежат на столе; кассир отошел от стола к сейфу."),
        ("unattended_cash_on_desk_on_exit", "Банкноты лежат на полу; кассир вышел из помещения."),
        ("counting_in_client_absence", "Кассир пересчитывает документы; клиент отсутствует."),
        ("safe_left_open_during_work", "Сейф открыт; банкноты лежат в ящике; кассир работает за столом."),
        ("collection_bag_accepted_without_check", "Инкассаторская сумка не принята кассиром; пломбу не проверили."),
    ):
        with pytest.raises(ValueError):
            validate_observation_consistency(bad_code, bad_obs)

    for good_code, good_obs in (
        ("personal_phone_usage", "Кассир использует смартфон за рабочим столом, сейф закрыт."),
        ("safe_left_open_during_work", "Кассир не использует сейф и сидит за столом: дверца сейфа открыта, внутри видны ценности."),
        ("personal_phone_usage", "Кассир использует смартфон; клиент держит телефон."),
        ("personal_phone_usage", "Кассир держит смартфон у уха; номер на бейдже неразличим."),
        ("collection_bag_accepted_without_check", "Кассир не проверил пломбу и принял инкассаторскую сумку."),
        ("personal_phone_usage", "Кассир не использует стационарный аппарат, а говорит по смартфону."),
        # Положительные контроли ревью 1017 (N02):
        ("personal_phone_usage", "Кассир держит смартфон в руке за рабочим столом."),
        ("unattended_cash_on_desk_on_exit", "Банкноты лежат на рабочем столе; кассир вышел из помещения."),
        ("counting_in_client_absence", "Кассир пересчитывает банкноты на столе; клиент вышел из помещения."),
        ("safe_left_open_during_work", "Дверца сейфа открыта и внутри сейфа видны банкноты; кассир работает за столом и не обращается к сейфу."),
        ("collection_bag_accepted_without_check", "Кассир принял инкассаторскую сумку; кассир не проверил пломбу и документы."),
    ):
        assert validate_observation_consistency(good_code, good_obs)

    # N01 (ревью 1017): Разделение уверенно чужого домена и рабочего/офисного помещения со столом
    for office_scene in (
        "В кадре показано офисное помещение с рабочим столом, а не банковская касса.",
        "Все кадры показывают рабочий кабинет сотрудника со столом и ноутбуком.",
    ):
        is_bank, dom_status, dom_reason = LocalVLMEngine._classify_scene_domain(office_scene, False)
        assert is_bank is True
        assert dom_status == "office_workstation_interior"
        assert dom_reason is None

    for foreign_scene in (
        "Уличная камера наблюдения: парковка перед зданием.",
        "Офисный коридор и лестничная площадка без рабочего стола.",
        "Учебная аудитория со студентами.",
    ):
        is_bank, dom_status, dom_reason = LocalVLMEngine._classify_scene_domain(foreign_scene, False)
        assert is_bank is False
        assert dom_status == "rejected_non_banking_domain"
        assert dom_reason is not None

    # Проверка сохранения физических булевых полей и condition_checks (0909 Section 4.2) + явного типа источника в cv_only (0909 Section 4.1)
    stub_frames = [
        DecodedFrame(f"F{i:04d}", i, i, float(i), float(i), 32, 32, np.zeros((32, 32, 3), dtype=np.uint8))
        for i in range(4)
    ]
    stub_bundle = VideoTimelineBundle(
        "s.mov", "s", 32, 32, 1.0, "1/1", 1.0, 0.0, "zero_based_pts", 4.0, False, 1.0, "coarse", stub_frames, {f.frame_id: f for f in stub_frames}
    )
    stub_ev = PyAVVideoDecoder.encode_verified_evidence_frames(stub_bundle, ["F0000", "F0001", "F0002"])
    safe_entry = format_violation_entry(
        "safe_left_open_during_work",
        1.0,
        stub_ev,
        "Сейф открыт и внутри видны деньги, пока кассир не использует сейф и печатает на компьютере.",
        structured_conditions={
            "safe_door_open": True,
            "safe_contents_visible": True,
            "cashier_left_room": False,
        },
    )
    assert safe_entry["evidence_source"] == "vlm_verified"
    assert safe_entry["safe_door_open"] is True
    assert safe_entry["safe_contents_visible"] is True
    assert safe_entry["cashier_left_room"] is False
    assert len(safe_entry["condition_checks"]) == 3

    cv_hyp_entry = format_violation_entry(
        "personal_phone_usage",
        1.0,
        stub_ev,
        "Сотрудник в рабочей зоне взаимодействует с мобильным телефоном (детекция смартфона вблизи рук на 5 кадрах окна).",
        evidence_source="cv_detector_unverified",
    )
    assert cv_hyp_entry["evidence_source"] == "cv_detector_unverified"
    assert cv_hyp_entry["condition_checks"][0]["status"] == "unverified_cv_hypothesis"
    assert cv_hyp_entry["mandatory_conditions_confirmed"] == []
    assert cv_hyp_entry["phone_in_hands"] is None

    # 4. R06 & 0909 (Section 4.3): Сохранение разных акторов и слияние одного непрерывного эпизода (включая порядковые числительные и episode_id)
    two_ep = CashDisciplineAnalyzer._merge_same_episode_events(
        [
            {"code": "personal_phone_usage", "timecode_sec": 9.0, "candidate_id": "LEFT", "window_idx": 0, "evidence_frame_ids": ["F0008", "F0009", "F0010"], "primary_evidence_frame_ids": ["F0009"], "observed_facts": "Работник слева держит смартфон."},
            {"code": "personal_phone_usage", "timecode_sec": 11.0, "candidate_id": "RIGHT", "window_idx": 0, "evidence_frame_ids": ["F0008", "F0011", "F0012"], "primary_evidence_frame_ids": ["F0011"], "observed_facts": "Работник справа держит смартфон."},
        ]
    )
    assert len(two_ep) == 2

    for follow_text in (
        "Кассир непрерывно использует смартфон за рабочим столом.",
        "Второй кассир за тем же столом всё ещё держит смартфон у уха.",
        "Первый сотрудник продолжает разговор по мобильному телефону.",
    ):
        one_ep = CashDisciplineAnalyzer._merge_same_episode_events(
            [
                {"code": "personal_phone_usage", "timecode_sec": 32.0, "candidate_id": "VLM_COARSE_01", "window_idx": 0, "evidence_frame_ids": ["F0032"], "primary_evidence_frame_ids": ["F0032"], "observed_facts": "Кассир за рабочим столом разговаривает по смартфону."},
                {"code": "personal_phone_usage", "timecode_sec": 32.0, "candidate_id": "VLM_COARSE_02", "window_idx": 1, "evidence_frame_ids": ["F0032"], "primary_evidence_frame_ids": ["F0032"], "observed_facts": follow_text},
            ]
        )
        assert len(one_ep) == 1

    ep_id_merged = CashDisciplineAnalyzer._merge_same_episode_events(
        [
            {"code": "personal_phone_usage", "timecode_sec": 30.0, "candidate_id": "VLM_COARSE_01", "window_idx": 0, "episode_id": "phone-episode-1", "actor_description": "Кассир в синей рубашке", "evidence_frame_ids": ["F0030"], "primary_evidence_frame_ids": ["F0030"], "observed_facts": "Кассир в синей рубашке подносит телефон к уху."},
            {"code": "personal_phone_usage", "timecode_sec": 32.0, "candidate_id": "VLM_COARSE_02", "window_idx": 1, "episode_id": "phone-episode-1", "actor_description": "Кассир в синей рубашке", "evidence_frame_ids": ["F0032"], "primary_evidence_frame_ids": ["F0032"], "observed_facts": "Кассир в синей рубашке продолжает разговор по телефону."},
        ]
    )
    assert len(ep_id_merged) == 1
    assert ep_id_merged[0]["episode_id"] == "phone-episode-1"
    assert ep_id_merged[0]["actor_description"] == "Кассир в синей рубашке"

    # 5. R07: Гарантированный верхний предел шага <= 3.5s в deep-режиме на сетке 0.75s
    grid_frames = [
        DecodedFrame(f"F{i:04d}", i, i, round(i * 0.75, 2), round(i * 0.75, 2), 64, 36, np.zeros((36, 64, 3), dtype=np.uint8))
        for i in range(60)
    ]
    grid_bundle = VideoTimelineBundle(
        "g.mov", "g", 64, 36, 1.33, "3/4", 1.33, 0.0, "zero_based_pts", 45.0, False, 0.75, "coarse", grid_frames, {f.frame_id: f for f in grid_frames}
    )
    deep_wins = CashDisciplineAnalyzer._build_coarse_overview_windows(grid_bundle, frames_per_window=12, max_interval_sec=3.5)
    for w in deep_wins:
        gaps = [b.timestamp_sec - a.timestamp_sec for a, b in zip(w, w[1:])]
        assert max(gaps) <= 3.5

    # 6. R10: Глобальное сопоставление минимальной стоимости не зависит от порядка GT
    labels = {"probe": {"official_verdict": "violation"}}
    anns_fwd = {"probe": {"origin_group_id": "p", "expected_events": [{"code": "personal_phone_usage", "timecode_sec": 1.9}, {"code": "personal_phone_usage", "timecode_sec": 3.0}]}}
    anns_rev = {"probe": {"origin_group_id": "p", "expected_events": [{"code": "personal_phone_usage", "timecode_sec": 3.0}, {"code": "personal_phone_usage", "timecode_sec": 1.9}]}}
    rec = [{"stem": "probe", "status": "done", "verdict": "violation", "processing_time_sec": 0.1, "violations": [{"code": "personal_phone_usage", "timecode_sec": 0.0}, {"code": "personal_phone_usage", "timecode_sec": 2.0}]}]
    m_fwd = compute_metrics_summary(rec, labels, anns_fwd)["temporal_localization_on_positives"]
    m_rev = compute_metrics_summary(rec, labels, anns_rev)["temporal_localization_on_positives"]
    assert m_fwd == m_rev
    assert m_fwd["within_2_0s_count"] == 2
    assert m_fwd["abs_errors_sec_list"] == [1.9, 1.0]

    # 7. R14 & R15: Реальное освобождение ёмкости очереди при истечении 2 ожидающих заданий во время работы активного и отсутствие зашитого API-ключа в index.html
    with tempfile.TemporaryDirectory() as q_storage:
        mgr = ProcessJobManager(storage_dir=q_storage, max_queue_size=2)
        try:
            first = mgr.submit_job("unused.mov", "blocking.mov", timeout_sec=10, simulate_sleep_sec=60)
            until = time.monotonic() + 3.0
            while mgr._jobs[first["job_id"]]["internal_state"] != "running" and time.monotonic() < until:
                time.sleep(0.01)
            q1 = mgr.submit_job("unused.mov", "queued1.mov", timeout_sec=0.25, simulate_sleep_sec=60)
            q2 = mgr.submit_job("unused.mov", "queued2.mov", timeout_sec=0.25, simulate_sleep_sec=60)
            time.sleep(0.8)
            for qj in (q1, q2):
                saved_q = json.load(open(os.path.join(q_storage, f"{qj['job_id']}.json"), "r", encoding="utf-8"))
                assert saved_q["internal_state"] == "failed_queue_timeout"
            assert mgr._pending_queue.qsize() == 0
            replacement = mgr.submit_job("unused.mov", "replacement.mov", timeout_sec=10, simulate_sleep_sec=60)
            assert replacement["status"] == "processing"
            assert replacement["internal_state"] == "queued"
        finally:
            mgr.shutdown()

    html_src = open(os.path.join(BASE_DIR, "app", "static", "index.html"), "r", encoding="utf-8").read()
    assert "forte-datathon26-key" not in html_src
    assert "analysis_incomplete" in html_src




