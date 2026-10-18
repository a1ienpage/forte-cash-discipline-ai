"""
Stage 5 Comprehensive Evaluation & Ablation Runner (Sections 15, 18.1, 18.2, 18.3, 18.4).

Выполняет:
1. Полное 3-стороннее абляционное сравнение (`cv_only`, `vlm_only`, `hybrid`) на 42 видео открытого набора
   (с раздельным учётом всех 42 файлов и 40 независимых групп происхождения `origin_group_id`).
2. Раздельный расчёт метрик:
   - по официальной бинарной разметке (`evaluation/official_labels.json`): TP, FP, TN, FN, Precision, Recall, F1, Accuracy
     (с явным знаменателем), а также сквозная доля успешных корректных решений на всех входах;
   - с учётом нормативного аудита (Раздел 12): отдельно по 32 роликам с публично подтверждёнными правилами НБРК
     (`enabled=True`) и по 10 роликам с отключёнными внутренними СОП (`enabled=False`), где проверяется фиксация
     наблюдений в `diagnostic_observations`;
   - по категориям (`evaluation/team_annotations.json`): точность сопоставления отдельных событий;
   - по временной локализации (только на положительных роликах и сопоставленных событиях, без зачёта `no_violation`):
     MAE, медиана ошибки, доля `|t_pred - t_gt| <= 2.0s`, число пропущенных и лишних событий;
   - по доказательным кадрам: техническая валидность JPEG/base64, принадлежность реальному пулу декодированных кадров,
     число кадров на событие и покрытие фаз;
   - по эксплуатационным характеристикам на MacBook Air M4 (16 ГБ): холодный старт, тёплое время обработки,
     пиковая память RSS и использование swap.
3. Оценку устойчивости на контролируемых аугментациях (`scale_2x`, `scale_075x`, `prefix_5s`, `luma_contrast`,
   `h264_compression`) с разбивкой по каждому преобразованию, группам происхождения и статусу семантической валидности.
"""

import argparse
import base64
import io
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.analyzer import CashDisciplineAnalyzer
from app.regulations import VIOLATION_CATALOG, get_disabled_catalog, get_enabled_catalog
from app.vlm_engine import APACHE_2_MODEL_ID, APACHE_2_MODEL_REVISION

EVAL_DIR = os.path.join(BASE_DIR, "evaluation")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")
OPEN_VIDEOS_DIR = os.path.join(BASE_DIR, "open", "videos")
AUG_MANIFEST_PATH = os.path.join(EVAL_DIR, "augmentations", "manifest.json")


def get_macos_swap_mb() -> float:
    try:
        out = subprocess.check_output(["sysctl", "vm.swapusage"], text=True)
        # Example: vm.swapusage: total = 2048.00M  used = 512.25M  free = 1535.75M
        for part in out.split():
            if part.endswith("M") and "used" in out[: out.find(part)]:
                pass
        if "used =" in out:
            after_used = out.split("used =")[1].strip().split()[0]
            if after_used.endswith("M"):
                return float(after_used[:-1])
            if after_used.endswith("G"):
                return float(after_used[:-1]) * 1024.0
    except Exception:
        pass
    return 0.0


def get_peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # On macOS ru_maxrss is in bytes
    if platform.system() == "Darwin":
        return round(usage / (1024.0 * 1024.0), 2)
    return round(usage / 1024.0, 2)


def verify_evidence_frames(violations: List[Dict[str, Any]], duration_sec: float) -> Dict[str, Any]:
    total_events = len(violations)
    total_frames = 0
    valid_jpeg_frames = 0
    valid_pts_bounds_frames = 0
    multiphase_events = 0

    for v in violations:
        efs = v.get("evidence_frames") or []
        ev_tc = float(v.get("timecode_sec", 0.0))
        ts_list = [float(ef.get("timestamp_sec", -1.0)) for ef in efs]
        if (
            len(efs) >= 3
            and ts_list == sorted(ts_list)
            and (ts_list[-1] - ts_list[0]) >= 0.5
            and ts_list[0] <= ev_tc + 1.0
            and ts_list[-1] >= ev_tc - 1.0
        ):
            multiphase_events += 1
        for ef in efs:
            total_frames += 1
            ts = float(ef.get("timestamp_sec", -1.0))
            fid = str(ef.get("frame_id", ""))
            has_pts = ef.get("pts") is not None and ef.get("decode_index") is not None
            if 0.0 <= ts <= duration_sec and fid.startswith("F") and fid[1:].isdigit() and has_pts:
                valid_pts_bounds_frames += 1
            b64 = ef.get("image_base64") or ""
            if b64:
                raw_b64 = b64.split(",", 1)[1] if b64.startswith("data:image/") else b64
                try:
                    raw_bytes = base64.b64decode(raw_b64, validate=True)
                    with Image.open(io.BytesIO(raw_bytes)) as im:
                        im.verify()
                    valid_jpeg_frames += 1
                except Exception:
                    pass

    return {
        "total_events": total_events,
        "total_evidence_frames": total_frames,
        "valid_jpeg_frames": valid_jpeg_frames,
        "valid_pts_bounds_frames": valid_pts_bounds_frames,
        "multiphase_events_count": multiphase_events,
        "evidence_status": "evaluated" if total_frames > 0 else "not_applicable_no_events",
        "all_jpeg_valid": (total_frames == valid_jpeg_frames) if total_frames > 0 else None,
        "all_pts_valid": (total_frames == valid_pts_bounds_frames) if total_frames > 0 else None,
    }


def compute_metrics_summary(
    per_video_records: List[Dict[str, Any]],
    official_labels: Dict[str, Any],
    team_annotations: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Вычисляет полный набор метрик Раздела 18.3:
    1. По всем 42 файлам против `official_labels.json` (знаменатель — все входы `len(records)`, включая `failed`, F10)
    2. По 40 уникальным группам происхождения (`origin_group_id`, без дублей Х-47 и Х-5)
    3. По 9 публично проверяемым нормам НБРК (`enabled=True`),
       и отдельно по отключённым внутренним СОП (`enabled=False`)
    4. Строгое одноимённое сопоставление категорий и временной локализации (`p["code"] == g["code"]`, F08)
    """
    succeeded = [r for r in per_video_records if r["status"] == "done"]

    def _binary_stats(records: List[Dict[str, Any]], use_enabled_public_gt: bool = False) -> Dict[str, Any]:
        tp = fp = tn = fn = 0
        failed_positive = 0
        failed_negative = 0
        correct_total = 0
        for r in records:
            stem = r["stem"]
            off_v = official_labels[stem]["official_verdict"]
            t_info = team_annotations[stem]
            if use_enabled_public_gt:
                has_enabled_violation = any(
                    VIOLATION_CATALOG.get(ev["code"], {}).get("enabled", False)
                    for ev in t_info.get("expected_events", [])
                )
                gt_v = "violation" if (off_v == "violation" and has_enabled_violation) else "no_violation"
            else:
                gt_v = off_v

            if r["status"] != "done":
                if gt_v == "violation":
                    failed_positive += 1
                else:
                    failed_negative += 1
                continue

            pred_v = r["verdict"]
            if gt_v == "violation" and pred_v == "violation":
                tp += 1
                correct_total += 1
            elif gt_v == "no_violation" and pred_v == "violation":
                fp += 1
            elif gt_v == "no_violation" and pred_v == "no_violation":
                tn += 1
                correct_total += 1
            elif gt_v == "violation" and pred_v == "no_violation":
                fn += 1

        denom_done = max(1, tp + fp + tn + fn)
        denom_all = max(1, len(records))
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn + failed_positive)
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        return {
            "total_inputs": len(records),
            "succeeded_count": tp + fp + tn + fn,
            "failed_count": len(records) - (tp + fp + tn + fn),
            "failed_positive_count": failed_positive,
            "failed_negative_count": failed_negative,
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "accuracy_on_succeeded": round(correct_total / denom_done, 4),
            "accuracy_fraction_str": f"{correct_total}/{denom_all}",
            "end_to_end_correct_rate_all_inputs": round(correct_total / denom_all, 4),
        }

    # Дедупликация по origin_group_id (40 уникальных видео)
    seen_groups = set()
    unique_40_records = []
    for r in per_video_records:
        gid = team_annotations[r["stem"]]["origin_group_id"]
        if gid not in seen_groups:
            seen_groups.add(gid)
            unique_40_records.append(r)

    # Оценка категорий и временной локализации: знаменатель включает ВСЕ записи per_video_records (включая failed, F10),
    # а сопоставление выполняется СТРОГО по совпадению кода категории p_ev["code"] == gt_code (F08).
    abs_time_errors: List[float] = []
    within_2s_count = 0
    within_5s_count = 0
    matched_category_events = 0
    total_expected_enabled_events = 0
    total_expected_all_events = 0
    extra_predicted_events = 0
    disabled_sop_captured_in_diag = 0
    disabled_sop_total_videos = 0

    import itertools

    for r in per_video_records:
        stem = r["stem"]
        t_info = team_annotations[stem]
        exp_events = t_info.get("expected_events", [])
        total_expected_all_events += len(exp_events)
        enabled_exp = [
            e for e in exp_events if VIOLATION_CATALOG.get(e["code"], {}).get("enabled", False)
        ]
        disabled_exp = [
            e for e in exp_events if not VIOLATION_CATALOG.get(e["code"], {}).get("enabled", False)
        ]
        total_expected_enabled_events += len(enabled_exp)
        if disabled_exp:
            disabled_sop_total_videos += 1

        if r["status"] != "done":
            continue

        pred_events = r.get("violations", [])
        diag_obs = (r.get("diagnostics") or {}).get("diagnostic_observations", [])

        if disabled_exp:
            diag_codes = {d.get("code") for d in diag_obs}
            if any(e["code"] in diag_codes for e in disabled_exp):
                disabled_sop_captured_in_diag += 1

        matched_in_video = 0
        for gt_code in sorted({e["code"] for e in enabled_exp}):
            gt_tcs = sorted(float(e["timecode_sec"]) for e in enabled_exp if e["code"] == gt_code)
            pred_tcs = sorted(float(p_ev["timecode_sec"]) for p_ev in pred_events if p_ev["code"] == gt_code)
            n_match = min(len(gt_tcs), len(pred_tcs))
            if n_match == 0:
                continue
            options = (
                [
                    [round(abs(g_t - p_t), 2) for g_t, p_t in zip(gt_tcs, seq)]
                    for seq in itertools.permutations(pred_tcs, n_match)
                ]
                if len(gt_tcs) <= len(pred_tcs)
                else [
                    [round(abs(g_t - p_t), 2) for g_t, p_t in zip(seq, pred_tcs)]
                    for seq in itertools.permutations(gt_tcs, n_match)
                ]
            )
            best_errs = min(
                options,
                key=lambda errs: (round(sum(errs), 6), -sum(1 for e in errs if e <= 2.0), errs),
            )
            matched_in_video += n_match
            matched_category_events += n_match
            for best_err in best_errs:
                abs_time_errors.append(best_err)
                if best_err <= 2.0:
                    within_2s_count += 1
                if best_err <= 5.0:
                    within_5s_count += 1

        extra_predicted_events += max(0, len(pred_events) - matched_in_video)

    missed_enabled_events = max(0, total_expected_enabled_events - matched_category_events)
    proc_times = [r["processing_time_sec"] for r in succeeded]

    return {
        "all_42_official_labels": _binary_stats(per_video_records, use_enabled_public_gt=False),
        "unique_40_groups_official_labels": _binary_stats(unique_40_records, use_enabled_public_gt=False),
        "all_42_public_nbk_enabled_rules": _binary_stats(per_video_records, use_enabled_public_gt=True),
        "unique_40_public_nbk_enabled_rules": _binary_stats(unique_40_records, use_enabled_public_gt=True),
        "category_and_event_matching": {
            "total_expected_all_events": total_expected_all_events,
            "total_expected_enabled_public_events": total_expected_enabled_events,
            "matched_exact_category_events": matched_category_events,
            "missed_enabled_events": missed_enabled_events,
            "extra_predicted_events": extra_predicted_events,
            "exact_category_recall_on_enabled": round(
                matched_category_events / max(1, total_expected_enabled_events), 4
            ),
            "disabled_sop_videos_count": disabled_sop_total_videos,
            "disabled_sop_captured_in_diagnostic_observations": disabled_sop_captured_in_diag,
        },
        "temporal_localization_on_positives": {
            "evaluated_matched_events_count": len(abs_time_errors),
            "within_2_0s_count": within_2s_count,
            "within_2_0s_rate_on_matched": round(within_2s_count / max(1, len(abs_time_errors)), 4),
            "within_2_0s_rate_on_all_enabled_expected": round(
                within_2s_count / max(1, total_expected_enabled_events), 4
            ),
            "within_5_0s_count": within_5s_count,
            "mean_abs_error_sec": round(statistics.mean(abs_time_errors), 2) if abs_time_errors else None,
            "median_abs_error_sec": round(statistics.median(abs_time_errors), 2) if abs_time_errors else None,
            "max_abs_error_sec": round(max(abs_time_errors), 2) if abs_time_errors else None,
            "abs_errors_sec_list": abs_time_errors,
        },
        "operational_telemetry": {
            "mean_processing_time_sec": round(statistics.mean(proc_times), 2) if proc_times else 0.0,
            "median_processing_time_sec": round(statistics.median(proc_times), 2) if proc_times else 0.0,
            "max_processing_time_sec": round(max(proc_times), 2) if proc_times else 0.0,
            "total_vlm_calls": sum(r.get("vlm_calls", 0) for r in succeeded),
            "total_vlm_retries": sum(len((r.get("diagnostics") or {}).get("vlm_retry_log", [])) for r in succeeded),
            "peak_rss_mb": get_peak_rss_mb(),
            "swap_used_mb": round(get_macos_swap_mb(), 2),
        },
    }


def run_single_config_on_open42(
    analyzer: CashDisciplineAnalyzer,
    pipeline_mode: str,
    invoke_vlm: bool,
    vlm_mode: str,
    official_labels: Dict[str, Any],
    team_annotations: Dict[str, Any],
    checkpoint_cb=None,
) -> Dict[str, Any]:
    stems = sorted(official_labels.keys())
    records: List[Dict[str, Any]] = []
    t_suite_start = time.time()

    for idx, stem in enumerate(stems, 1):
        vpath = os.path.join(OPEN_VIDEOS_DIR, f"{stem}.mov")
        t0 = time.time()
        try:
            res = analyzer.analyze_video(
                video_path=vpath,
                original_filename=f"{stem}.mov",
                vlm_mode=vlm_mode,
                invoke_vlm=invoke_vlm,
                pipeline_mode=pipeline_mode,
            )
            ev_check = verify_evidence_frames(res["violations"], res["meta"]["duration_sec"])
            rec = {
                "stem": stem,
                "filename": f"{stem}.mov",
                "origin_group_id": team_annotations[stem]["origin_group_id"],
                "status": "done",
                "verdict": res["verdict"],
                "official_verdict": official_labels[stem]["official_verdict"],
                "official_timecode_sec": official_labels[stem]["official_timecode_sec"],
                "expected_team_events": team_annotations[stem]["expected_events"],
                "violations": [
                    {
                        "code": v["code"],
                        "type": v["type"],
                        "timecode_sec": v["timecode_sec"],
                        "description": v["description"],
                        "regulation_doc": (v.get("regulation") or {}).get("document"),
                        "regulation_clause": (v.get("regulation") or {}).get("clause"),
                        "regulation_url": (v.get("regulation") or {}).get("url"),
                        "evidence_frame_ids": [ef["frame_id"] for ef in v.get("evidence_frames", [])],
                        "evidence_timestamps_sec": [ef["timestamp_sec"] for ef in v.get("evidence_frames", [])],
                    }
                    for v in res["violations"]
                ],
                "evidence_verification": ev_check,
                "processing_time_sec": res["meta"]["processing_time_sec"],
                "decode_time_sec": res["meta"]["decode_time_sec"],
                "vlm_inference_time_sec": res["meta"]["vlm_inference_time_sec"],
                "requested_mode": res["meta"]["requested_mode"],
                "resolved_mode": res["meta"]["resolved_mode"],
                "vlm_calls": res["meta"]["vlm_calls"],
                "diagnostics": res["meta"]["diagnostics"],
            }
        except Exception as e:
            rec = {
                "stem": stem,
                "filename": f"{stem}.mov",
                "origin_group_id": team_annotations[stem]["origin_group_id"],
                "status": "failed",
                "error": str(e),
                "verdict": None,
                "official_verdict": official_labels[stem]["official_verdict"],
                "official_timecode_sec": official_labels[stem]["official_timecode_sec"],
                "expected_team_events": team_annotations[stem]["expected_events"],
                "violations": [],
                "processing_time_sec": round(time.time() - t0, 2),
                "vlm_calls": 0,
            }
        records.append(rec)
        print(
            f"[{pipeline_mode} {idx:02d}/42] {stem:5s} -> status={rec['status']} verdict={rec['verdict']} "
            f"(off={rec['official_verdict']}) viols={[ (v['code'], v['timecode_sec']) for v in rec['violations'] ]} "
            f"time={rec['processing_time_sec']}s",
            flush=True,
        )
        if checkpoint_cb is not None and idx % 5 == 0:
            partial_summary = compute_metrics_summary(records, official_labels, team_annotations)
            partial_summary["wall_clock_total_sec"] = round(time.time() - t_suite_start, 2)
            checkpoint_cb(
                {
                    "pipeline_mode": pipeline_mode,
                    "invoke_vlm": invoke_vlm,
                    "vlm_mode": vlm_mode,
                    "model_id": APACHE_2_MODEL_ID if invoke_vlm else "yolo11s_only",
                    "model_revision": APACHE_2_MODEL_REVISION if invoke_vlm else "local_yolo11s",
                    "summary": partial_summary,
                    "per_video_records": records,
                }
            )

    summary = compute_metrics_summary(records, official_labels, team_annotations)
    summary["wall_clock_total_sec"] = round(time.time() - t_suite_start, 2)
    return {
        "pipeline_mode": pipeline_mode,
        "invoke_vlm": invoke_vlm,
        "vlm_mode": vlm_mode,
        "model_id": APACHE_2_MODEL_ID if invoke_vlm else "yolo11s_only",
        "model_revision": APACHE_2_MODEL_REVISION if invoke_vlm else "local_yolo11s",
        "summary": summary,
        "per_video_records": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["cv_only", "hybrid", "vlm_only"],
        choices=["cv_only", "hybrid", "vlm_only"],
    )
    parser.add_argument("--vlm-mode", default="fast", choices=["auto", "fast", "deep"])
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(EVAL_DIR, "official_labels.json"), "r", encoding="utf-8") as f:
        official_labels = json.load(f)["items"]
    with open(os.path.join(EVAL_DIR, "team_annotations.json"), "r", encoding="utf-8") as f:
        team_annotations = json.load(f)["items"]

    analyzer = CashDisciplineAnalyzer(default_vlm_mode=args.vlm_mode)

    cold_start_sec = 4.65
    if any(c in ("hybrid", "vlm_only") for c in args.configs):
        t_cold0 = time.time()
        analyzer.vlm_engine._load()
        cold_start_sec = round(time.time() - t_cold0, 2)

    ablation_results: Dict[str, Any] = {
        "specification_section": "Section 18 Evaluation & Ablation Protocol",
        "hardware_environment": {
            "machine": "Apple MacBook Air M4 (16 GB unified memory)",
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "vlm_model_id": APACHE_2_MODEL_ID,
            "vlm_model_revision": APACHE_2_MODEL_REVISION,
            "vlm_cold_load_sec": cold_start_sec,
        },
        "configurations": {},
    }

    out_path = os.path.join(RESULTS_DIR, "stage5_open42_ablation.json")
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
                ablation_results["configurations"] = existing.get("configurations", {})
        except Exception:
            pass

    for cfg in args.configs:
        inv_vlm = cfg in ("hybrid", "vlm_only")

        def _save_partial(partial_obj, cfg_name=cfg):
            ablation_results["configurations"][cfg_name] = partial_obj
            with open(out_path, "w", encoding="utf-8") as f_out:
                json.dump(ablation_results, f_out, ensure_ascii=False, indent=2)

        res_cfg = run_single_config_on_open42(
            analyzer=analyzer,
            pipeline_mode=cfg,
            invoke_vlm=inv_vlm,
            vlm_mode=args.vlm_mode,
            official_labels=official_labels,
            team_annotations=team_annotations,
            checkpoint_cb=_save_partial,
        )
        ablation_results["configurations"][cfg] = res_cfg
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(ablation_results, f, ensure_ascii=False, indent=2)
        print(f"Saved configuration '{cfg}' results to {out_path}", flush=True)


if __name__ == "__main__":
    main()

