"""
Stage 5 Controlled Augmentations Evaluation Runner (Sections 15.1–15.4, 18.3, 18.4).

Оценивает устойчивость конвейера на контролируемых аугментациях (`scale_2x`, `scale_075x`, `prefix_5s`,
`luma_contrast`, `h264_compression`):
- изменение бинарного вердикта и категории относительно семантически валидного исходника;
- точность сдвига временной локализации `t_new = t_orig + delta_measured` на `prefix_5s` (доля `<= 2.0s`, MAE);
- раздельная агрегация по каждому из 5 типов преобразований и по независимым группам происхождения `origin_group_id`;
- явный учёт вариантов со статусами `degraded_fine_detail_warning` и `illumination_semantics_check`.
"""

import argparse
import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.analyzer import CashDisciplineAnalyzer
from app.regulations import VIOLATION_CATALOG

EVAL_DIR = os.path.join(BASE_DIR, "evaluation")
AUG_MANIFEST_PATH = os.path.join(EVAL_DIR, "augmentations", "manifest.json")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")


def evaluate_augmentations(
    analyzer: CashDisciplineAnalyzer,
    manifest_entries: List[Dict[str, Any]],
    source_results_by_stem: Dict[str, Dict[str, Any]],
    pipeline_mode: str,
    invoke_vlm: bool,
    vlm_mode: str,
    pilot_only: bool = False,
    checkpoint_cb=None,
) -> Dict[str, Any]:
    entries = [e for e in manifest_entries if (not pilot_only or e.get("is_pilot_10"))]
    records: List[Dict[str, Any]] = []

    def _aggregate(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
        by_aug_type: Dict[str, Any] = {}
        for atype in ["scale_2x", "scale_075x", "prefix_5s", "luma_contrast", "h264_compression"]:
            sub = [r for r in recs if r["augmentation_type"] == atype]
            # Знаменатель включает ВСЕ семантически валидные входы (включая любые failed, F10)
            strict_valid = [r for r in sub if r["semantic_validity"] == "valid"]
            strict_valid_done = [r for r in strict_valid if r["status"] == "done"]
            strict_valid_failed = [r for r in strict_valid if r["status"] != "done"]
            flagged_variants = [r for r in sub if r["semantic_validity"] != "valid"]
            verdict_matches = sum(1 for r in strict_valid if r["status"] == "done" and r["verdict_consistent_with_source"])
            cat_matches = sum(1 for r in strict_valid if r["status"] == "done" and r["category_consistent_with_source"])
            shift_errs = [e for r in strict_valid_done for e in r["time_shift_errors_vs_source_sec"]]
            source_events_total = sum(len(r.get("source_violations", [])) for r in strict_valid)
            missed_source_events = sum(int(r.get("missed_source_events_count", 0)) for r in strict_valid)
            extra_pred_events = sum(int(r.get("extra_predicted_events_count", 0)) for r in strict_valid_done)
            within_2s = sum(1 for e in shift_errs if e <= 2.0)
            unique_groups = len({r["origin_group_id"] for r in sub})

            by_aug_type[atype] = {
                "total_files": len(sub),
                "unique_origin_groups": unique_groups,
                "strict_semantically_valid_count": len(strict_valid),
                "strict_valid_succeeded_count": len(strict_valid_done),
                "strict_valid_failed_count": len(strict_valid_failed),
                "flagged_degraded_or_illumination_count": len(flagged_variants),
                "flagged_files": [
                    {
                        "filename": r["augmented_filename"],
                        "semantic_validity": r["semantic_validity"],
                        "reason": r["metric_exclusion_reason"],
                        "augmented_verdict": r["augmented_verdict"],
                    }
                    for r in flagged_variants
                ],
                "verdict_consistency_rate": round(verdict_matches / max(1, len(strict_valid)), 4),
                "verdict_consistency_fraction": f"{verdict_matches}/{len(strict_valid)}",
                "category_consistency_rate": round(cat_matches / max(1, len(strict_valid)), 4),
                "category_consistency_fraction": f"{cat_matches}/{len(strict_valid)}",
                "source_events_total": source_events_total,
                "temporal_shift_evaluated_events": len(shift_errs),
                "missed_source_events_count": missed_source_events,
                "extra_predicted_events_count": extra_pred_events,
                "temporal_shift_within_2_0s_count": within_2s,
                "temporal_shift_within_2_0s_rate": round(within_2s / max(1, len(shift_errs)), 4) if shift_errs else None,
                "temporal_shift_within_2_0s_rate_on_all_source_events": round(
                    within_2s / max(1, source_events_total), 4
                ) if source_events_total > 0 else None,
                "temporal_shift_mean_abs_error_sec": round(statistics.mean(shift_errs), 2) if shift_errs else None,
            }

        return {
            "pipeline_mode": pipeline_mode,
            "pilot_only": pilot_only,
            "total_evaluated_files": len(recs),
            "unique_origin_groups": len({r["origin_group_id"] for r in recs}),
            "by_augmentation_type": by_aug_type,
            "records": recs,
        }

    for idx, entry in enumerate(entries, 1):
        raw_apath = entry["augmented_path"]
        apath = raw_apath if os.path.isabs(raw_apath) else os.path.join(BASE_DIR, raw_apath)
        stem = entry["source_stem"]
        aug_type = entry["augmentation_type"]
        shift_sec = float(entry.get("timeline_shift_sec", 0.0))
        src_rec = source_results_by_stem.get(stem, {})
        src_verdict = src_rec.get("verdict")
        src_viols = src_rec.get("violations", [])

        t0 = time.time()
        try:
            res = analyzer.analyze_video(
                video_path=apath,
                original_filename=entry["augmented_filename"],
                vlm_mode=vlm_mode,
                invoke_vlm=invoke_vlm,
                pipeline_mode=pipeline_mode,
            )
            pred_v = res["verdict"]
            pred_viols = [
                {
                    "code": v["code"],
                    "timecode_sec": v["timecode_sec"],
                }
                for v in res["violations"]
            ]
            status = "done"
            err_msg = None
            proc_sec = res["meta"]["processing_time_sec"]
            vlm_calls = res["meta"]["vlm_calls"]
        except Exception as e:
            pred_v = None
            pred_viols = []
            status = "failed"
            err_msg = str(e)
            proc_sec = round(time.time() - t0, 2)
            vlm_calls = 0

        verdict_consistent_with_source = (status == "done" and pred_v == src_verdict)
        src_codes = [v["code"] for v in src_viols]
        pred_codes = [v["code"] for v in pred_viols]
        category_consistent_with_source = (status == "done" and set(src_codes) == set(pred_codes))

        # Строгое сопоставление один-к-одному ТОЛЬКО для событий с одинаковым кодом категории (F09, R10)
        import itertools

        time_shift_errors_vs_source: List[float] = []
        matched_source_events_count = 0
        if status == "done":
            for code in sorted({sv["code"] for sv in src_viols}):
                s_tcs = sorted(float(sv["timecode_sec"]) + shift_sec for sv in src_viols if sv["code"] == code)
                p_tcs = sorted(float(pv["timecode_sec"]) for pv in pred_viols if pv["code"] == code)
                n_match = min(len(s_tcs), len(p_tcs))
                if n_match == 0:
                    continue
                options = (
                    [
                        [round(abs(s_t - p_t), 2) for s_t, p_t in zip(s_tcs, seq)]
                        for seq in itertools.permutations(p_tcs, n_match)
                    ]
                    if len(s_tcs) <= len(p_tcs)
                    else [
                        [round(abs(s_t - p_t), 2) for s_t, p_t in zip(seq, p_tcs)]
                        for seq in itertools.permutations(s_tcs, n_match)
                    ]
                )
                best_errs = min(
                    options,
                    key=lambda errs: (round(sum(errs), 6), -sum(1 for e in errs if e <= 2.0), errs),
                )
                matched_source_events_count += n_match
                time_shift_errors_vs_source.extend(best_errs)

        missed_source_events_count = max(0, len(src_viols) - matched_source_events_count)
        extra_predicted_events_count = max(0, len(pred_viols) - matched_source_events_count) if status == "done" else 0

        time_errors_vs_gt: List[float] = []
        if status == "done":
            enabled_gt_evs = [
                ev_gt
                for ev_gt in entry.get("expected_team_events", [])
                if VIOLATION_CATALOG.get(ev_gt["code"], {}).get("enabled", False)
            ]
            for code in sorted({ev_gt["code"] for ev_gt in enabled_gt_evs}):
                g_tcs = sorted(float(ev_gt["timecode_sec"]) for ev_gt in enabled_gt_evs if ev_gt["code"] == code)
                p_tcs = sorted(float(pv["timecode_sec"]) for pv in pred_viols if pv["code"] == code)
                n_match = min(len(g_tcs), len(p_tcs))
                if n_match == 0:
                    continue
                options = (
                    [
                        [round(abs(g_t - p_t), 2) for g_t, p_t in zip(g_tcs, seq)]
                        for seq in itertools.permutations(p_tcs, n_match)
                    ]
                    if len(g_tcs) <= len(p_tcs)
                    else [
                        [round(abs(g_t - p_t), 2) for g_t, p_t in zip(seq, p_tcs)]
                        for seq in itertools.permutations(g_tcs, n_match)
                    ]
                )
                best_gt_errs = min(
                    options,
                    key=lambda errs: (round(sum(errs), 6), -sum(1 for e in errs if e <= 2.0), errs),
                )
                time_errors_vs_gt.extend(best_gt_errs)

        rec = {
            "augmented_filename": entry["augmented_filename"],
            "source_stem": stem,
            "origin_group_id": entry["origin_group_id"],
            "is_pilot_10": entry.get("is_pilot_10", False),
            "augmentation_type": aug_type,
            "semantic_validity": entry.get("semantic_validity", "valid"),
            "metric_exclusion_reason": entry.get("metric_exclusion_reason"),
            "timeline_shift_sec": shift_sec,
            "status": status,
            "error": err_msg,
            "source_verdict": src_verdict,
            "augmented_verdict": pred_v,
            "expected_official_verdict": entry["expected_official_verdict"],
            "verdict_consistent_with_source": verdict_consistent_with_source,
            "category_consistent_with_source": category_consistent_with_source,
            "source_violations": [{"code": v["code"], "timecode_sec": v["timecode_sec"]} for v in src_viols],
            "augmented_violations": pred_viols,
            "time_shift_errors_vs_source_sec": time_shift_errors_vs_source,
            "missed_source_events_count": missed_source_events_count,
            "extra_predicted_events_count": extra_predicted_events_count,
            "time_errors_vs_gt_sec": time_errors_vs_gt,
            "processing_time_sec": proc_sec,
            "vlm_calls": vlm_calls,
        }
        records.append(rec)
        print(
            f"[AUG-{pipeline_mode} {idx:03d}/{len(entries):03d}] {entry['augmented_filename']:30s} -> "
            f"verdict={pred_v} (src={src_verdict}) consist={verdict_consistent_with_source} "
            f"viols={pred_viols} shift_err={time_shift_errors_vs_source}",
            flush=True,
        )
        if checkpoint_cb is not None and idx % 5 == 0:
            checkpoint_cb(_aggregate(records))

    return _aggregate(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm-mode", default="fast", choices=["auto", "fast", "deep"])
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(AUG_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    entries = manifest["entries"]

    with open(os.path.join(RESULTS_DIR, "stage5_open42_ablation.json"), "r", encoding="utf-8") as f:
        open42_data = json.load(f)["configurations"]

    analyzer = CashDisciplineAnalyzer(default_vlm_mode=args.vlm_mode)
    out_path = os.path.join(RESULTS_DIR, "stage5_augmentations_eval.json")

    payload: Dict[str, Any] = {
        "specification_section": "Section 15 & Section 18.3 Controlled Augmentations Evaluation",
        "manifest_total_files": len(entries),
        "cv_only_all_210": {},
        "hybrid_pilot_50": {},
    }

    # 1. Полный прогон cv_only на всех 210 аугментированных файлах (все 42 исходника × 5 преобразований)
    cv_src_map = {r["stem"]: r for r in open42_data["cv_only"]["per_video_records"]}
    cv_aug_res = evaluate_augmentations(
        analyzer=analyzer,
        manifest_entries=entries,
        source_results_by_stem=cv_src_map,
        pipeline_mode="cv_only",
        invoke_vlm=False,
        vlm_mode=args.vlm_mode,
        pilot_only=False,
    )
    payload["cv_only_all_210"] = cv_aug_res
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # 2. Полный прогон гибридного конвейера (CV + VLM) на 50 аугментированных файлах пилотной выборки (10 исходников × 5 преобразований)
    hyb_src_map = {r["stem"]: r for r in open42_data.get("hybrid", open42_data["cv_only"])["per_video_records"]}

    def _save_hyb_partial(partial_hyb):
        payload["hybrid_pilot_50"] = partial_hyb
        with open(out_path, "w", encoding="utf-8") as f_out:
            json.dump(payload, f_out, ensure_ascii=False, indent=2)

    hyb_aug_res = evaluate_augmentations(
        analyzer=analyzer,
        manifest_entries=entries,
        source_results_by_stem=hyb_src_map,
        pipeline_mode="hybrid",
        invoke_vlm=True,
        vlm_mode=args.vlm_mode,
        pilot_only=True,
        checkpoint_cb=_save_hyb_partial,
    )
    payload["hybrid_pilot_50"] = hyb_aug_res
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved controlled augmentations evaluation to {out_path}", flush=True)


if __name__ == "__main__":
    main()
