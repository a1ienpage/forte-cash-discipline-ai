"""
Stage 1: Фиксация исходного состояния (Baseline Snapshot), контрольных сумм открытого датасета
и строгое разделение официальной разметки (`official_labels.json`) и собственной разметки команды (`team_annotations.json`).

В соответствии с Разделом 3 ТЗ на доработку:
- Официальный набор (`open/open_labels.xlsx`) содержит только бинарный вердикт и строковый таймстемп без категорий нарушений.
- Собственная разметка команды хранится отдельно в `evaluation/team_annotations.json` с указанием процедуры подготовки,
  групп дубликатов и неоднозначных нормативных/визуальных случаев.
- Производственный код `app/` НЕ импортирует и НЕ читает файлы из `evaluation/`.
"""

import glob
import hashlib
import json
import os
import re
import subprocess
from typing import Dict, Any, List, Optional

import openpyxl

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(BASE_DIR, "evaluation")
SNAP_DIR = os.path.join(EVAL_DIR, "baseline_snapshot")
OPEN_VIDEOS_DIR = os.path.join(BASE_DIR, "open", "videos")
EXCEL_PATH = os.path.join(BASE_DIR, "open", "open_labels.xlsx")


def parse_russian_timestamp(ts_raw: Optional[str]) -> Optional[float]:
    if not ts_raw or not str(ts_raw).strip():
        return None
    s = str(ts_raw).strip().lower()
    m_min_sec = re.search(r"(\d+)\s*минут[аы]?\s*(\d+)\s*секунд", s)
    if m_min_sec:
        return float(int(m_min_sec.group(1)) * 60 + int(m_min_sec.group(2)))
    m_sec = re.search(r"(\d+(?:\.\d+)?)\s*секунд", s)
    if m_sec:
        return float(m_sec.group(1))
    m_num = re.search(r"(\d+(?:\.\d+)?)", s)
    if m_num:
        return float(m_num.group(1))
    return None


def probe_video_ffprobe(path: str) -> Dict[str, Any]:
    cmd = [
        "/opt/homebrew/bin/ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,time_base,start_pts,start_time,duration,nb_frames",
        "-of",
        "json",
        path,
    ]
    out = subprocess.check_output(cmd, text=True)
    data = json.loads(out)
    streams = data.get("streams", [{}])
    return streams[0] if streams else {}


def main():
    os.makedirs(SNAP_DIR, exist_ok=True)

    # 1. Контрольные суммы и технический профиль 42 файлов
    video_files = sorted(glob.glob(os.path.join(OPEN_VIDEOS_DIR, "*.mov")))
    sha_to_files: Dict[str, List[str]] = {}
    records: List[Dict[str, Any]] = []

    for vp in video_files:
        fname = os.path.basename(vp)
        stem = os.path.splitext(fname)[0]
        with open(vp, "rb") as f:
            raw = f.read()
        md5_h = hashlib.md5(raw).hexdigest()
        sha256_h = hashlib.sha256(raw).hexdigest()
        sha_to_files.setdefault(sha256_h, []).append(stem)
        st_info = probe_video_ffprobe(vp)
        records.append(
            {
                "filename": fname,
                "stem": stem,
                "size_bytes": len(raw),
                "md5": md5_h,
                "sha256": sha256_h,
                "stream_info": st_info,
            }
        )

    # Присвоение origin_group_id (одинаковый SHA256 -> одна группа происхождения)
    sha_to_group: Dict[str, str] = {}
    for idx, (sha, stems) in enumerate(sorted(sha_to_files.items(), key=lambda x: x[1][0]), 1):
        sha_to_group[sha] = f"open_group_{idx:02d}_{'_'.join(stems)}"

    duplicate_pairs = [stems for stems in sha_to_files.values() if len(stems) > 1]
    for r in records:
        r["origin_group_id"] = sha_to_group[r["sha256"]]
        r["duplicates"] = [s for s in sha_to_files[r["sha256"]] if s != r["stem"]]

    checksums_payload = {
        "dataset_name": "ForteBank Datathon 26 Open Dataset",
        "total_files": len(records),
        "unique_sha256_groups": len(sha_to_files),
        "exact_duplicate_groups": duplicate_pairs,
        "hash_usage_policy": "Checksums are recorded strictly for data provenance, duplicate grouping, and reproducibility. Prohibitions per Section 3 & 5.1: never used by app/ to select or influence analysis results.",
        "files": records,
    }
    with open(os.path.join(EVAL_DIR, "dataset_checksums.json"), "w", encoding="utf-8") as f:
        json.dump(checksums_payload, f, ensure_ascii=False, indent=2)

    # 2. Официальная разметка из open_labels.xlsx (без категорий)
    wb = openpyxl.load_workbook(EXCEL_PATH)
    ws = wb.active
    official_items: Dict[str, Dict[str, Any]] = {}
    for row_idx, row in enumerate(ws.iter_rows(values_only=True)):
        if row_idx == 0 or not row or not row[0]:
            continue
        stem = str(row[0]).strip()
        ts_raw = row[1]
        verdict_raw = str(row[2]).strip().lower() if row[2] else ""
        verdict = "violation" if verdict_raw == "нарушение" else "no_violation"
        ts_sec = parse_russian_timestamp(ts_raw)
        official_items[stem] = {
            "stem": stem,
            "official_verdict": verdict,
            "official_timestamp_raw": ts_raw,
            "official_timecode_sec": ts_sec,
        }

    official_payload = {
        "source_file": "open/open_labels.xlsx",
        "description": "Официальная разметка открытого набора от организаторов дататона. Содержит только бинарный вердикт и таймкод без категорий нарушений и нормативных ссылок.",
        "total_entries": len(official_items),
        "items": official_items,
    }
    with open(os.path.join(EVAL_DIR, "official_labels.json"), "w", encoding="utf-8") as f:
        json.dump(official_payload, f, ensure_ascii=False, indent=2)

    # 3. Собственная разметка команды (team_annotations.json) с указанием методологии и неоднозначных случаев
    team_categories = {
        "X-1": {
            "team_category_code": "post_count_cash_manipulation",
            "expected_events": [{"code": "post_count_cash_manipulation", "timecode_sec": 22.0}],
            "regulatory_basis_status": "internal_sop_or_indirect_nbk231",
            "ambiguity_note": "На 22с кассир отделяет несколько купюр вручную после пересчёта на машинке. Прямого запрета в тексте Постановления НБРК № 231 (п. 19–20) на отделение части купюр без повторного прогона всей пачки через счётчик нет (требуется соответствие фактической суммы приходному документу); правило опирается на внутренний регламент кассового контроля банка.",
        },
        "X-2": {
            "team_category_code": "no_id_or_manual_recount",
            "expected_events": [{"code": "no_id_or_manual_recount", "timecode_sec": 10.0}],
            "regulatory_basis_status": "ambiguous_visibility_and_procedure",
            "ambiguity_note": "Клиент кладёт пачку денег на стол, кассир помещает их в счётчик. Отсутствие бумажного удостоверения в кадре не доказывает непроверку личности (п. 19 пп. 1 НБРК № 231 допускает данные из сервиса цифровых документов в терминале АБИС, кроме того проверка могла быть выполнена до начала фрагмента). Машинный пересчёт на счётчике сам по себе является полистным пересчётом.",
        },
        "Х-4": {
            "team_category_code": "unattended_previous_client_cash",
            "expected_events": [
                {"code": "unattended_previous_client_cash", "timecode_sec": 57.0},
                {"code": "unattended_previous_client_cash", "timecode_sec": 68.0},
            ],
            "regulatory_basis_status": "public_mandatory_nbk231_clause20",
            "ambiguity_note": "Файл побайтово идентичен Х-47 (SHA256 совпадает). В официальной таблице у Х-4 указана 57 секунда (вход нового клиента при неубранных деньгах первого клиента), а у Х-47 — 68 секунда (1 мин 8 сек, приём денег второго клиента рядом с деньгами первого). Прямое нарушение п. 20 абз. 2 Правил НБРК № 231.",
        },
        "Х-11": {
            "team_category_code": "closing_procedure_lights_off_return",
            "expected_events": [{"code": "closing_procedure_lights_off_return", "timecode_sec": 18.0}],
            "regulatory_basis_status": "internal_sop_only",
            "ambiguity_note": "Кассир выключает свет на 17-18с и возвращается к столу в темноте. В публичных Постановлениях НБРК № 231 и № 14 отсутствует норма о порядке выключения света и запрете возврата к столу после выключения освещения — это правило внутреннего распорядка банка.",
        },
        "Х-13": {
            "team_category_code": "collection_bag_stored_without_recount",
            "expected_events": [{"code": "collection_bag_stored_without_recount", "timecode_sec": 47.0}],
            "regulatory_basis_status": "ambiguous_stage_nbk231",
            "ambiguity_note": "Кассир извлекает корешки из зелёной инкассаторской сумки и на 47с убирает их в сейф без пересчёта на счётчике. В п. 24–25 Правил НБРК № 231 описан приём сумок на хранение; порядок и сроки вскрытия и полистного пересчёта инкассированной выручки определяются внутренними правилами банка.",
        },
        "Х-15": {
            "team_category_code": "unattended_collection_bag_on_desk",
            "expected_events": [{"code": "unattended_collection_bag_on_desk", "timecode_sec": 15.0}],
            "regulatory_basis_status": "public_mandatory_nbk231_clause20_80",
            "ambiguity_note": "Кассир покидает кассу на 15с, оставив инкассаторскую сумку открыто на рабочем столе (п. 20 абз. 2, п. 80 Правил НБРК № 231).",
        },
        "Х-17": {
            "team_category_code": "collection_bag_accepted_without_check",
            "expected_events": [{"code": "collection_bag_accepted_without_check", "timecode_sec": 7.0}],
            "regulatory_basis_status": "public_mandatory_nbk231_clause24",
            "ambiguity_note": "Быстрая передача инкассаторской сумки из рук в руки на 7с без осмотра целостности, пломбы и сверки с препроводительным документом (п. 24 Правил НБРК № 231).",
        },
        "Х-18": {
            "team_category_code": "multiple_clients_in_cash_room",
            "expected_events": [{"code": "multiple_clients_in_cash_room", "timecode_sec": 20.0}],
            "regulatory_basis_status": "public_mandatory_nbk231_clause20_and_nbk14",
            "ambiguity_note": "На 20с во время работы с наличными первого клиента в кассу входит второй человек и передаёт предметы через плечо первого клиента (п. 20 Правил НБРК № 231, п. 16 Правил НБРК № 14).",
        },
        "Х-20": {
            "team_category_code": "safe_left_open_during_work",
            "expected_events": [{"code": "safe_left_open_during_work", "timecode_sec": 5.0}],
            "regulatory_basis_status": "public_mandatory_nbk231_clause20_80",
            "ambiguity_note": "Кассир берёт наличные из сейфа на 5с и работает за столом при распахнутой дверце сейфа (п. 20 абз. 2, п. 80 Правил НБРК № 231).",
        },
        "Х-22": {
            "team_category_code": "safe_left_open_on_exit",
            "expected_events": [{"code": "safe_left_open_on_exit", "timecode_sec": 14.0}],
            "regulatory_basis_status": "public_mandatory_nbk231_clause20_80",
            "ambiguity_note": "При выходе из помещения кассы на 14с дверца сейфа остаётся приоткрытой/незапертой (п. 20 абз. 2, п. 80 Правил НБРК № 231).",
        },
        "Х-24": {
            "team_category_code": "unattended_documents_stamps_on_exit",
            "expected_events": [{"code": "unattended_documents_stamps_on_exit", "timecode_sec": 8.0}],
            "regulatory_basis_status": "public_mandatory_nbk231_clause80",
            "ambiguity_note": "При выходе кассира на 8с на столе остаются папка с документами и кассовые принадлежности/штампы (п. 80 Правил НБРК № 231).",
        },
        "Х-26": {
            "team_category_code": "unattended_cash_on_desk_on_exit",
            "expected_events": [{"code": "unattended_cash_on_desk_on_exit", "timecode_sec": 9.0}],
            "regulatory_basis_status": "public_mandatory_nbk231_clause20",
            "ambiguity_note": "Кассир оставляет пачку банкнот на столе и выходит из помещения кассы на 9с (п. 20 абз. 2 Правил НБРК № 231).",
        },
        "Х-28": {
            "team_category_code": "unlocked_workstation_on_exit",
            "expected_events": [{"code": "unlocked_workstation_on_exit", "timecode_sec": 7.0}],
            "regulatory_basis_status": "internal_it_security_sop",
            "ambiguity_note": "Кассир выходит из кассы на 7с при открытом ноутбуке со светящимся экраном. В Постановлении НБРК № 231 нет нормы о закрытии крышки ноутбука или блокировке ОС при выходе из кассы, а по общему плану камеры невозможно достоверно отличить активную сессию АБИС от заставки блокировки.",
        },
        "Х-30": {
            "team_category_code": "unauthorized_person_in_cashier_zone",
            "expected_events": [{"code": "unauthorized_person_in_cashier_zone", "timecode_sec": 13.0}],
            "regulatory_basis_status": "public_mandatory_nbk14_clause8_16",
            "ambiguity_note": "На 13с во внутреннюю зону кассира заходит второй человек через входную дверь (п. 8, п. 16 Правил НБРК № 14). Примечание: визуально без бейджа/контекста невозможно определить должность вошедшего сотрудника.",
        },
        "Х-32": {
            "team_category_code": "cash_door_left_unlocked_on_entry",
            "expected_events": [{"code": "cash_door_left_unlocked_on_entry", "timecode_sec": 5.0}],
            "regulatory_basis_status": "ambiguous_visual_lock_state",
            "ambiguity_note": "Кассир входит в кассу и садится за стол на 5с, не совершая движения запирания замка изнутри. При закрытом полотне двери и наличии электромагнитного/автоматического замка факт незапертости двери визуально недоказуем без проверки ручки или видимой щели.",
        },
        "Х-33": {
            "team_category_code": "personal_phone_usage",
            "expected_events": [{"code": "personal_phone_usage", "timecode_sec": 6.0}],
            "regulatory_basis_status": "public_mandatory_nbk231_clause12",
            "ambiguity_note": "На 6с кассир садится за рабочий стол и держит/использует смартфон в руках (п. 12 абз. 2 Правил НБРК № 231 запрещает внос работниками в кассовый узел средств связи с функцией фото/видеофиксации).",
        },
        "Х-34": {
            "team_category_code": "counting_in_client_absence",
            "expected_events": [{"code": "counting_in_client_absence", "timecode_sec": 13.0}],
            "regulatory_basis_status": "public_mandatory_nbk231_clause20",
            "ambiguity_note": "На 13с клиент встаёт со стула и выходит из кассы, пока кассир продолжает пересчёт принятых наличных на счётчике (п. 20 абз. 1 Правил НБРК № 231: приём и пересчёт осуществляются в присутствии клиента).",
        },
        "Х-38": {
            "team_category_code": "multiple_clients_in_cash_room",
            "expected_events": [{"code": "multiple_clients_in_cash_room", "timecode_sec": 33.0}],
            "regulatory_basis_status": "public_mandatory_nbk231_clause20",
            "ambiguity_note": "На 33с после ухода первого клиента входит следующий клиент, когда деньги предыдущего клиента ещё лежат на столе и убираются в сейф при новом клиенте (п. 20 абз. 2 Правил НБРК № 231).",
        },
        "Х-40": {
            "team_category_code": "personal_belongings_outerwear_on_desk",
            "expected_events": [{"code": "personal_belongings_outerwear_on_desk", "timecode_sec": 17.0}],
            "regulatory_basis_status": "public_mandatory_nbk231_clause12",
            "ambiguity_note": "На 17с кассир снимает верхнюю одежду (куртку) и кладёт её вместе с личной сумкой на рабочий стол в кассе (п. 12 абз. 1 Правил НБРК № 231: хранение личных вещей работников осуществляется вне кассового узла).",
        },
        "Х-44": {
            "team_category_code": "mixed_denominations_packaging",
            "expected_events": [{"code": "mixed_denominations_packaging", "timecode_sec": 16.0}],
            "regulatory_basis_status": "public_mandatory_nbk231_clause43_46",
            "ambiguity_note": "На 16с после пересчёта кассир складывает купюры разных номиналов в одну стопку. Примечание: по разрешению 464x848 номинал отдельных банкнот различим ограниченно (только по цветовой гамме и раздельным стопкам до объединения).",
        },
        "Х-46": {
            "team_category_code": "food_or_drink_at_workstation",
            "expected_events": [{"code": "food_or_drink_at_workstation", "timecode_sec": 11.0}],
            "regulatory_basis_status": "internal_sop_and_nbk231_clause12",
            "ambiguity_note": "На 11с кассир берёт стакан с напитком и пьёт за рабочим столом рядом с наличными. В Постановлении НБРК № 231 нет прямого упоминания напитков/еды (есть только общее требование п. 12 о хранении личных вещей вне кассового узла и п. 3 пп. 6 о защите банкнот от повреждений); прямой запрет напитков относится к внутреннему СОП.",
        },
        "Х-47": {
            "team_category_code": "unattended_previous_client_cash",
            "expected_events": [
                {"code": "unattended_previous_client_cash", "timecode_sec": 57.0},
                {"code": "unattended_previous_client_cash", "timecode_sec": 68.0},
            ],
            "regulatory_basis_status": "public_mandatory_nbk231_clause20",
            "ambiguity_note": "Побайтовый дубликат файла Х-4. В официальной разметке указан таймстемп 1 минута 8 секунда (68.0с).",
        },
    }

    team_items: Dict[str, Any] = {}
    for r in records:
        stem = r["stem"]
        off = official_items.get(stem, {})
        t_cat = team_categories.get(stem)
        team_items[stem] = {
            "stem": stem,
            "origin_group_id": r["origin_group_id"],
            "duplicates": r["duplicates"],
            "official_verdict": off.get("official_verdict", "no_violation"),
            "official_timecode_sec": off.get("official_timecode_sec"),
            "team_category_code": t_cat["team_category_code"] if t_cat else None,
            "expected_events": t_cat["expected_events"] if t_cat else [],
            "regulatory_basis_status": t_cat["regulatory_basis_status"] if t_cat else "clean_reference",
            "ambiguity_note": t_cat["ambiguity_note"] if t_cat else "Визуально подтверждённых нарушений в ролике не зафиксировано.",
        }

    team_payload = {
        "author_and_methodology": (
            "Разметка подготовлена инженерной командой Datathon 26 путём ручного покадрового просмотра 42 открытых видеороликов "
            "и сверки наблюдаемых действий с официальными текстами Постановления Правления НБРК № 231 (V1900019680) и № 14 (V2000020075). "
            "Официальный файл open/open_labels.xlsx не содержит категорий нарушений, поэтому оценка точности категорий возможна "
            "исключительно относительно данной собственной разметки команды с учётом задокументированных неоднозначных случаев."
        ),
        "duplicate_pairs_note": (
            "В 42 видеофайлах обнаружены 2 пары полных побайтовых дубликатов (40 уникальных видео): "
            "1) Х-4 и Х-47 (нарушение, в официальной таблице размечены разные фазы одного длящегося эпизода: 57с и 68с); "
            "2) Х-8 и Х-10 (оба без нарушений)."
        ),
        "items": team_items,
    }
    with open(os.path.join(EVAL_DIR, "team_annotations.json"), "w", encoding="utf-8") as f:
        json.dump(team_payload, f, ensure_ascii=False, indent=2)

    # 4. Метаданные базового снимка (baseline_metadata.json)
    baseline_meta = {
        "snapshot_timestamp": "2026-10-17T22:26:00+05:00",
        "purpose": "Baseline snapshot prior to engineering specification v1.0 refactoring (Section 3 & Section 20 Stage 1).",
        "hardware": {
            "machine": "MacBook Air Apple M4",
            "ram_gb": 16,
            "storage_gb": 512,
        },
        "model": {
            "repo_id": "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
            "snapshot_revision": "fdcc572e8b05ba9daeaf71be8c9e4267c826ff9b",
            "license": "Apache-2.0",
        },
        "pre_refactor_environment": {
            "python": "3.13.9",
            "mlx": "0.31.1",
            "mlx_vlm": "0.7.8",
            "pydantic": "2.12.3",
            "fastapi": "0.143.3",
            "opencv_python": "5.0.0",
            "ultralytics": "8.4.178",
            "transformers": "5.20.1",
        },
        "known_pre_refactor_issues_recorded": [
            "CVSceneAnalyzer and CashDisciplineAnalyzer relied on scene_layout classes (A, B1, B2, C1, C2), hardcoded pixel slices, duration thresholds, and pre-written descriptions.",
            "Frame timestamps used frame_index / fps via cv2.VideoCapture instead of PyAV PTS * time_base.",
            "VLM prompt instructed model to confirm CV signals instead of treating them as unverified hypotheses.",
            "requirements.txt had stale version pins conflicting with installed packages.",
            "Some rules in regulations.py relied on internal bank SOPs or visually unprovable conditions without enabled/disabled flags.",
            "Main server used ThreadPoolExecutor without hard process termination upon timeout.",
        ],
    }
    with open(os.path.join(SNAP_DIR, "baseline_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(baseline_meta, f, ensure_ascii=False, indent=2)

    print("Stage 1 completed: official_labels.json, team_annotations.json, dataset_checksums.json, baseline_metadata.json created.")


if __name__ == "__main__":
    main()
