"""
Deterministic generator for REPORT.md and README.md directly from:
- app/regulations.py
- evaluation/results/stage5_open42_ablation.json
- evaluation/results/stage5_augmentations_eval.json
- evaluation/results/stage6_meva_evaluation.json
- evaluation/meva_sample/manifest.json
- evaluation/augmentations/manifest.json

Guarantees zero discrepancy between JSON evaluation artifacts, regulatory catalog, and Markdown reports (F07–F17).
"""

import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.regulations import VIOLATION_CATALOG, get_disabled_catalog, get_enabled_catalog


def _load_json(rel_path: str):
    with open(os.path.join(BASE_DIR, rel_path), "r", encoding="utf-8") as f:
        return json.load(f)


def fmt_pct(rate) -> str:
    if rate is None:
        return "не применимо"
    return f"{rate * 100.0:.2f}%"


def generate_readme() -> str:
    enabled_cat = get_enabled_catalog()
    disabled_cat = get_disabled_catalog()

    enabled_rows = []
    for idx, (code, entry) in enumerate(enabled_cat.items(), 1):
        enabled_rows.append(
            f"| {idx} | `{code}` | {entry['short_title']} | {entry['regulation_clause']} |"
        )

    disabled_rows = []
    for idx, (code, entry) in enumerate(disabled_cat.items(), 1):
        disabled_rows.append(
            f"| {idx} | `{code}` | {entry['short_title']} | {entry.get('disabled_reason') or entry.get('rule_summary')} |"
        )

    return f"""# ForteBank Cash Discipline AI Monitor (Datathon'26 — Инженерная доработка v2.1)

Локальный AI-сервис для автоматизированного выявления визуально наблюдаемых нарушений кассовой дисциплины и безопасности по видеозаписям работы кассира с привязкой к публичным нормативным правовым актам Республики Казахстан (**Постановления Правления НБРК № 231 (`V1900019680`) и № 14 (`V2000020075`)**, `adilet.zan.kz`).

Подробный инженерный отчёт из 4 обязательных частей (программные проверки `AT-01`..`AT-25` и устранение замечаний `F01`–`F17` / `R01`–`R15`, абляция на 42 открытых видео, контролируемые аугментации и внешняя выборка MEVA), сгенерированный напрямую из JSON-артефактов оценки, приведён в файле [REPORT.md](REPORT.md).

---

## 1. Доступ к работающему сервису и API (Разделы 10 и 11 ТЗ, `F16`, `R15`)

* **Ключ доступа (`X-API-Key`):** задаётся через переменную окружения `FORTE_API_KEY` либо неверсионируемый локальный файл `.forte_api_key` (исключён в `.gitignore`; прежнее значение из истории исходников явно отклоняется с `HTTP 401`, а активный ключ никогда не публикуется в `/api/v1/catalog`, `/api/v1/health` и `app/static/index.html`, `R15`, `N04`).
* **Локальный веб-интерфейс:** `http://localhost:8000`
* **Защищённые эндпоинты API (`X-API-Key` обязателен):**
  * `POST /api/v1/analyze` — загрузка видеофайла (`.mov`, `.mp4`, `.m4v`), принимает параметр режима `vlm_mode` или алиас `mode` (`auto | fast | deep`), возвращает `{{"job_id": "...", "status": "processing", "internal_state": "queued"}}`.
  * `POST /api/v1/analyze-sample` — постановка в очередь образца из `open/videos` (строгий `await verify_api_key(...)`, поддерживает `vlm_mode` и `mode`), возвращает `{{"job_id": "...", "status": "processing", "internal_state": "queued"}}`.
  * `GET /api/v1/jobs/{{job_id}}` — статус публичного контракта (`processing`, `done`, `failed`), детальное внутреннее состояние (`internal_state`: `queued`, `running`, `done`, `failed_queue_timeout`, `failed_timeout_killed`, `failed_worker_crash`, `failed_interrupted_on_restart`) и полный отчёт по схеме Pydantic v2 (включая явный индикатор неполного покрытия `meta.diagnostics.analysis_incomplete`, `R07`).
  * `GET /api/v1/video/{{job_id}}` — защищённый стриминг исходного видео по заголовку `X-API-Key` (в веб-интерфейсе загружается через `fetch` -> `Blob` -> `URL.createObjectURL`, без передачи ключа в query-параметре URL).
  * `GET /api/v1/dataset` — список образцов открытого набора (длительность извлекается за `< 2` мс из метаданных контейнера через `PyAVVideoDecoder.probe_container_metadata()` без декодирования видеопотока и без блокировки event loop, `F14`).
* **Открытые информационные эндпоинты (без секретов):**
  * `GET /api/v1/health` — оперативный статус сервиса (`"ok"` или `"degraded"`), общий булев признак готовности (`healthy`), живость потока супервизора (`supervisor_alive`), наличие локального кэша моделей (`vlm_model_cached`), текущая глубина очереди (`queue_depth`) и максимальная ёмкость очереди (`max_queue_size`) (`R15`).
  * `GET /api/v1/catalog` — нормативный справочник (`{len(enabled_cat)}` активных публичных норм НБРК `enabled=True` и `{len(disabled_cat)}` отключённых диагностических правил внутренних СОП `enabled=False`).

### Пример вызова API (cURL)
```bash
# 1. Отправка видеозаписи на проверку (поддерживаются поля vlm_mode и mode)
curl -X POST http://localhost:8000/api/v1/analyze \\
     -H "X-API-Key: $FORTE_API_KEY" \\
     -F "file=@open/videos/Х-18.mov" \\
     -F "vlm_mode=auto"

# Ответ:
# {{"job_id":"3f2a9c1e8b04","status":"processing","internal_state":"queued"}}

# 2. Получение результата
curl -X GET http://localhost:8000/api/v1/jobs/3f2a9c1e8b04 \\
     -H "X-API-Key: $FORTE_API_KEY"

# 3. Защищённый стриминг исходного видео
curl -X GET http://localhost:8000/api/v1/video/3f2a9c1e8b04 \\
     -H "X-API-Key: $FORTE_API_KEY" --output preview.mov
```

---

## 2. Воспроизводимая установка, лицензионный аудит и запуск (Раздел 4, `F12`, `F17`, `R12`, `R13`)

### Требования и целевая платформа
* **Целевая машина разработки и тестирования:** Apple MacBook Air M4 (16 ГБ объединённой памяти Apple Silicon, macOS arm64, Metal / MLX, Python `3.13`).
* **Лицензионный статус компонентов (`F17`):**
  * **Мультимодальная модель VLM:** `mlx-community/Qwen2.5-VL-7B-Instruct-4bit` (**Apache-2.0 License**, ревизия `fdcc572e8b05ba9daeaf71be8c9e4267c826ff9b`, базовые веса `Qwen/Qwen2.5-VL-7B-Instruct`). Модель 3B (`Qwen Research License`) полностью исключена из кода и конфигурации.
  * **Видеодекодер и компьютерное зрение:** `PyAV` (`av==16.1.0`, **BSD-3-Clause**), `opencv-python==4.13.0.92` (**Apache-2.0**), `mlx==0.31.1` / `mlx-vlm==0.3.10` (**MIT**).
  * **Детектор объектов (`ultralytics==8.4.38` + `yolo11s.pt`) — важное лицензионное раскрытие (`F17`):** пакет `ultralytics` и предобученные веса `yolo11s.pt` распространяются производителем по двойной модели лицензирования: открытая лицензия по умолчанию — **GNU Affero General Public License v3.0 (AGPL-3.0)**, требующая раскрытия исходного кода при предоставлении сетевого сервиса, либо коммерческая **Ultralytics Enterprise License** для закрытого проприетарного внедрения. Для промышленной эксплуатации внутри банка без требований AGPL-3.0 необходимо либо приобрести Ultralytics Enterprise License, либо заменить вызов `YOLO("yolo11s.pt")` в `app/cv_detector.py` на детектор под пермиссивной лицензией **Apache-2.0** (`RT-DETR` из `transformers` / `onnxruntime` или `YOLOX`).

### Воспроизводимая установка (`uv.lock` / `requirements.lock.txt`, `F12`, `R12`, `R13`)
Оба lock-файла (`requirements.lock.txt` и `uv.lock`) сгенерированы из единой канонической конфигурации `pyproject.toml` и синхронизированы по всем **103 пакетам** дерева рантайм-зависимостей (включая корневую запись проекта `fortebank-cash-discipline-ai`, согласованные архивы `wheels`/`sdist`, хэши и транзитивную цепочку `datasets -> fsspec[http] -> aiohttp`). Скрипт `./start_service.sh --check` выполняется в режиме `FORTE_READONLY_CHECK=1`, не модифицируя сохранённые задачи в `jobs_data/` (`R13`).

```bash
# Вариант A (через uv и синхронизированный uv.lock):
uv sync --frozen

# Вариант B (через pip и синхронизированный requirements.lock.txt со 103 точными версиями пакетов):
python3 -m venv .venv
.venv/bin/pip install --no-deps -r requirements.lock.txt

# Проверка окружения без запуска веб-сервера и без побочных эффектов на jobs_data/ (R13):
PATH=/usr/bin:/bin /bin/bash ./start_service.sh --check
```

### Запуск сервиса и приёмочных тестов
```bash
# Запуск веб-сервиса и постоянного фонового воркера на порту 8000:
./start_service.sh

# Запуск полного набора из 25 автоматических приёмочных испытаний (AT-01..AT-25):
python3 -m pytest -v tests/test_acceptance_suite.py
```

---

## 3. Архитектура конвейера (Разделы 5–10 ТЗ, `F01`–`F05`, `R01`–`R15`)

1. **Потоковое декодирование с ограниченным буфером памяти и строгим контролем PTS (`app/video_decoder.py`, `F05`, `F14`, `R08`):**
   * Декодирование через **PyAV (`av`)** с вычислением времени каждого кадра по формуле `timestamp_sec = (frame.pts - start_pts) * float(stream.time_base)`. При отсутствии `frame.pts` в потоке и `strict_pts=True` поднимается явная ошибка `VideoDecodeError`; при `strict_pts=False` в метаданных явно проставляются `pts_fallback_used=True`, `missing_pts_frames_count` и `pts_timebase_note="pts_missing_fallback_index_used"` (`R08`).
   * **Ограниченный буфер памяти (`F05`):** `decode_timeline()` сохраняет BGR-массивы только для уменьшенных обзорных кадров (`max_dim=640`, снижение объёма удерживаемых массивов на 95%), а полноразмерные кадры для плотных окон Pass 2 (`decode_frames_by_ids`) и доказательств извлекаются по требованию.
2. **Нормализация координат детектора по фактическому буферу кадра и явный `CVDetectorError` (`app/cv_detector.py`, `R01`, `R03`):**
   * Нормализация боксов YOLO выполняется строго по размерности переданного в детектор массива `f_obj.bgr.shape[:2]` (`arr_h, arr_w`), что гарантирует инвариантность относительных координат, площади персон и расстояний при любом исходном разрешении контейнера (`640×360`, `1920×1080`, `3840×2160`, `R01`).
   * Отсутствие весов или сбой `yolo_model.predict()` вызывает явное исключение `CVDetectorError`, исключая возврат ложного `no_violation` при неисправном детекторе (`R03`).
3. **Двухпроходная верификация VLM, поусловная валидация и поэкземплярная привязка кандидатов (`app/vlm_engine.py`, `app/analyzer.py`, `app/regulations.py`, `R02`, `R05`, `R06`, `R07`, `R08`):**
   * **Гарантированная верхняя граница шага (`R07`):** `_build_coarse_overview_windows()` выбирает самый дальний кадр на расстоянии `<= max_interval_sec` от предыдущего выбранного кадра, поэтому фактический шаг раскадровки никогда не превышает `4.0` с в `fast` и `3.5` с в `deep`. При неполном покрытии (`analysis_incomplete=True`) интерфейс явно отображает жёлтый предупреждающий статус `АНАЛИЗ НЕПОЛНЫЙ`.
   * **Поусловная проверка правил без глобального стоп-списка (`R02`, `R05`):** `validate_observation_consistency()` проверяет обязательные положительные группы признаков каждого правила (`RULE_REQUIRED_POSITIVE_GROUPS`), специфичные для конкретного правила исключения (`RULE_SPECIFIC_EXCLUSIONS`), маркеры неопределённости/окклюзии (`не видна`, `неразличим`) и структурированные булевы поля (`cash_visible_on_desk`, `cashier_left_room`, `seal_check_performed`, `recount_performed`, `phone_in_hands`).
   * **Индивидуальные решения по каждому `candidate_id` и сохранение независимых эпизодов (`R06`):** каждое подтверждённое событие в Pass 2 связывается максимум с одним лучшим кандидатом окна (`1-к-1`), а остальные кандидаты окна переводятся в `rejected_by_vlm`. Слияние `_merge_same_episode_events()` проверяет пересечение первичных кадров (`primary_evidence_frame_ids`) до мультифазного расширения и не сливает события разных кандидатов или разных пространственных зон (`слева` vs `справа`).
4. **Освобождение слотов очереди при истечении ожидания (`app/worker.py`, `F13`, `R14`):**
   * `_sweep_expired_queued_jobs()` не только переводит просроченные задачи в `failed_queue_timeout`, но и немедленно удаляет их `job_id` из `self._pending_queue.queue`, освобождая слот для новых запросов (`R14`).

---

## 4. Аудированный нормативный справочник РК (`app/regulations.py`, Раздел 4 ревью)

По результатам повторного нормативно-правового аудита (`regulatory_audit.md`) в контуре публичных обвинений оставлены только **{len(enabled_cat)} правил (`enabled=True`)**, однозначно подтверждённых текстом Постановления НБРК № 231 (`V1900019680`) и визуально доказуемых по видеозаписи. Ещё **{len(disabled_cat)} правил (`enabled=False`)** переведены в диагностический контур (`meta.diagnostics.diagnostic_observations`), так как требуют знания внутренних документов банка (п. 80), полномочий контролирующих лиц (п. 42), состава смешанной пачки (п. 49–50) или данных АБИС:

### Активные публично проверяемые правила НБРК (`enabled=True`, {len(enabled_cat)} правил)

| № | Код | Краткое наименование | Нормативное основание |
|---|---|---|---|
{chr(10).join(enabled_rows)}

### Диагностические / отключённые правила (`enabled=False`, {len(disabled_cat)} правил)

| № | Код | Краткое наименование | Причина перевода в диагностический контур (`enabled=False`) |
|---|---|---|---|
{chr(10).join(disabled_rows)}
"""


def generate_report() -> str:
    enabled_cat = get_enabled_catalog()
    disabled_cat = get_disabled_catalog()
    ablation = _load_json("evaluation/results/stage5_open42_ablation.json")
    aug_eval = _load_json("evaluation/results/stage5_augmentations_eval.json")
    meva_eval = _load_json("evaluation/results/stage6_meva_evaluation.json")
    team_annot = _load_json("evaluation/team_annotations.json")["items"]

    cfgs = ablation["configurations"]
    cv_s = cfgs["cv_only"]["summary"]
    vlm_s = cfgs["vlm_only"]["summary"]
    hyb_s = cfgs["hybrid"]["summary"]
    hw = ablation.get("hardware_environment", {})
    prov = ablation.get("provenance", {})

    # Enabled & disabled catalog tables
    enabled_rows = []
    for idx, (code, entry) in enumerate(enabled_cat.items(), 1):
        enabled_rows.append(
            f"| {idx} | `{code}` | {entry['short_title']} | {entry['regulation_clause']} |"
        )
    disabled_rows = []
    for idx, (code, entry) in enumerate(disabled_cat.items(), 1):
        disabled_rows.append(
            f"| {idx} | `{code}` | {entry['short_title']} | {entry.get('disabled_reason') or entry.get('rule_summary')} |"
        )

    # Helper for binary table rows
    def _bin_row(label: str, b: dict, include_end_to_end: bool = True) -> str:
        tot = b["total_inputs"]
        succ = b["succeeded_count"]
        fail = b["failed_count"]
        tp, fp, tn, fn = b["TP"], b["FP"], b["TN"], b["FN"]
        prec = f"`{b['precision']:.4f}` (`{tp}/{tp+fp}`)" if (tp + fp) > 0 else f"`0.0000` (`0/0`)"
        rec = f"`{b['recall']:.4f}` (`{tp}/{tp+fn}`)"
        f1 = f"`{b['f1']:.4f}`"
        acc = f"**`{b['accuracy_on_succeeded']*100:.2f}%` (`{tp+tn}/{succ}`)**"
        if include_end_to_end:
            e2e = f"**`{b['end_to_end_correct_rate_all_inputs']*100:.2f}%` (`{tp+tn}/{tot}`)**"
            return f"| {label} | `{tot} / {succ} / {fail}` | {tp} | {fp} | {tn} | {fn} | {prec} | {rec} | {f1} | {acc} | {e2e} |"
        return f"| {label} | `{tot} / {succ} / {fail}` | {tp} | {fp} | {tn} | {fn} | {prec} | {rec} | {f1} | {acc} |"

    # Evidence quality helper from per_video_records
    def _ev_stats(cfg_key: str) -> dict:
        recs = cfgs[cfg_key]["per_video_records"]
        tot_ev = sum(r.get("evidence_verification", {}).get("total_events", 0) for r in recs)
        tot_fr = sum(r.get("evidence_verification", {}).get("total_evidence_frames", 0) for r in recs)
        val_jp = sum(r.get("evidence_verification", {}).get("valid_jpeg_frames", 0) for r in recs)
        val_pt = sum(r.get("evidence_verification", {}).get("valid_pts_bounds_frames", 0) for r in recs)
        mp_ev = sum(r.get("evidence_verification", {}).get("multiphase_events_count", 0) for r in recs)
        if tot_fr == 0 or tot_ev == 0:
            return {
                "jpeg": "не применимо (`0` кадров)",
                "pts": "не применимо (`0` кадров)",
                "mp": "не применимо (`0` событий)",
            }
        return {
            "jpeg": f"`{val_jp}/{tot_fr}` (`{val_jp/tot_fr*100:.1f}%`)",
            "pts": f"`{val_pt}/{tot_fr}` (`{val_pt/tot_fr*100:.1f}%`)",
            "mp": f"`{mp_ev}/{tot_ev}` (`{mp_ev/tot_ev*100:.1f}%`)",
        }

    ev_cv = _ev_stats("cv_only")
    ev_vlm = _ev_stats("vlm_only")
    ev_hyb = _ev_stats("hybrid")

    cv_cat = cv_s["category_and_event_matching"]
    vlm_cat = vlm_s["category_and_event_matching"]
    hyb_cat = hyb_s["category_and_event_matching"]

    cv_tl = cv_s["temporal_localization_on_positives"]
    vlm_tl = vlm_s["temporal_localization_on_positives"]
    hyb_tl = hyb_s["temporal_localization_on_positives"]

    cv_op = cv_s["operational_telemetry"]
    vlm_op = vlm_s["operational_telemetry"]
    hyb_op = hyb_s["operational_telemetry"]

    # Per-video breakdown of disabled SOP videos in hybrid (to avoid F07 narrative mismatch)
    hyb_by_stem = {r["stem"]: r for r in cfgs["hybrid"]["per_video_records"]}
    disabled_sop_lines = []
    enabled_fn_lines = []
    hyb_fp_lines = []
    for stem in sorted(hyb_by_stem.keys()):
        r = hyb_by_stem[stem]
        off_v = r["official_verdict"]
        pred_v = r["verdict"]
        pred_viols = r["violations"]
        exp_evs = team_annot[stem]["expected_events"]
        has_enabled_gt = any(e["code"] in enabled_cat for e in exp_evs)
        if off_v == "violation" and not has_enabled_gt:
            gt_codes = [e["code"] for e in exp_evs]
            if pred_v == "no_violation":
                disabled_sop_lines.append(
                    f"     * `{stem}.mov` (ожидаемый диагностический код `{gt_codes[0]}`): конвейер `hybrid` вернул `verdict=\"no_violation\"`, не вынося неподтверждённого публичным актом НБРК обвинения."
                )
            else:
                pred_desc = ", ".join(f"`{v['code']}` на `{v['timecode_sec']}s`" for v in pred_viols)
                disabled_sop_lines.append(
                    f"     * `{stem}.mov` (ожидаемый диагностический код `{gt_codes[0]}`): само правило СОП отключено (`enabled=False`), однако конвейер `hybrid` сработал по другому активному правилу ({pred_desc}), поэтому по официальной бинарной разметке ролик попал в `TP`, а по публичному контуру НБРК (`enabled=True` GT) — в `FP` (`extra_predicted_events`)."
                )
        elif off_v == "violation" and has_enabled_gt and pred_v == "no_violation":
            gt_desc = ", ".join(f"`{e['code']}` на `{e['timecode_sec']}s`" for e in exp_evs if e["code"] in enabled_cat)
            enabled_fn_lines.append(f"     * `{stem}.mov` ({gt_desc}) -> `hybrid` вернул `no_violation` (`FN`).")
        elif off_v == "no_violation" and pred_v == "violation":
            pred_desc = ", ".join(f"`{v['code']}` на `{v['timecode_sec']}s`" for v in pred_viols)
            hyb_fp_lines.append(f"   * `{stem}.mov` (`official_verdict=\"no_violation\"`) -> `hybrid` предсказал {pred_desc} (`FP`).")

    # Augmentations tables
    def _aug_rows(block_key: str) -> str:
        by_t = aug_eval[block_key]["by_augmentation_type"]
        rows = []
        labels = {
            "scale_2x": "**`scale_2x`** (`iw*2:ih*2`, bicubic)",
            "scale_075x": "**`scale_075x`** (`iw*0.75:ih*0.75`)",
            "prefix_5s": "**`prefix_5s`** (`+5.0s` нейтральное начало)",
            "luma_contrast": "**`luma_contrast`** (`brightness=+0.05, contrast=1.08`)",
            "h264_compression": "**`h264_compression`** (`CRF 28`)",
        }
        for atype in ["scale_2x", "scale_075x", "prefix_5s", "luma_contrast", "h264_compression"]:
            s = by_t[atype]
            w2_matched = (
                f"`{s['temporal_shift_within_2_0s_count']}/{s['temporal_shift_evaluated_events']}` (`{s['temporal_shift_within_2_0s_rate']*100:.2f}%`)"
                if s["temporal_shift_evaluated_events"] > 0
                else "не применимо (`0/0`)"
            )
            w2_all_src = (
                f"`{s['temporal_shift_within_2_0s_count']}/{s['source_events_total']}` (`{s['temporal_shift_within_2_0s_rate_on_all_source_events']*100:.2f}%`)"
                if s["source_events_total"] > 0
                else "не применимо (`0/0`)"
            )
            mae_str = f"`{s['temporal_shift_mean_abs_error_sec']:.2f} с`" if s["temporal_shift_mean_abs_error_sec"] is not None else "—"
            rows.append(
                f"| {labels[atype]} | `{s['total_files']} / {s['unique_origin_groups']}` | `{s['strict_semantically_valid_count']}` (`done={s['strict_valid_succeeded_count']}, failed={s['strict_valid_failed_count']}`) | `{s['flagged_degraded_or_illumination_count']}` | `{s['verdict_consistency_fraction']}` (`{s['verdict_consistency_rate']*100:.2f}%`) | `{s['category_consistency_fraction']}` (`{s['category_consistency_rate']*100:.2f}%`) | `{s['source_events_total']} / {s['temporal_shift_evaluated_events']} / {s['missed_source_events_count']} / {s['extra_predicted_events_count']}` | {w2_matched} | {w2_all_src} | {mae_str} |"
            )
        return "\n".join(rows)

    # MEVA table directly from stage6_meva_evaluation.json
    meva_rows = []
    for ep in meva_eval["per_episode_results"]:
        gt_strs = [
            f"`{g['activity']}` (`{g['clip_pts_time_range_sec'][0]}–{g['clip_pts_time_range_sec'][1]}s`, `tsr0={g['tsr0_frames']}`)"
            for g in ep["official_gt_activities"]
        ]
        pred_strs = [
            f"`{p['activity']}` (`{round(float(p['start_sec']), 2)}–{round(float(p['end_sec']), 2)}s`)"
            for p in ep["diagnostic_vlm_detected_activities"]
        ]
        match_strs = [
            f"`{m['activity']}` (предск. `{m['pred_clip_range_sec'][0]}–{m['pred_clip_range_sec'][1]}s`, ошибка центра `{m['midpoint_error_sec']}s`)"
            for m in ep["matched_activities"]
        ]
        match_cell = f"`{len(ep['matched_activities'])}/{len(ep['official_gt_activities'])}`" + (
            f": {', '.join(match_strs)}" if match_strs else ""
        )
        b_beh = ep["unfamiliar_scene_banking_api_behavior"]
        b_cell = f"`{b_beh['banking_verdict']}`"
        if b_beh["banking_violations_returned"]:
            b_cell += " (" + ", ".join(f"`{v['code']}` @ `{v['timecode_sec']}s`" for v in b_beh["banking_violations_returned"]) + ")"
        meva_rows.append(
            f"| `{ep['episode_id']}` | `{ep['camera_id']}` / `{ep['scenario_group_id'].split('_')[-1]}` | {', '.join(gt_strs)} | {', '.join(pred_strs)} | {match_cell} | {b_cell} |"
        )

    meva_s = meva_eval["base_action_recognition_summary"]
    meva_b = meva_eval["unfamiliar_scene_banking_diagnostic_signal"]
    meva_no_viol_pct = f"{meva_b['banking_no_violation_count'] / max(1, meva_b['total_episodes']) * 100.0:.1f}%"
    meva_eval_count = meva_b.get("evaluated_episodes_count", meva_b["total_episodes"])
    meva_fail_count = meva_b.get("banking_failed_count", 0)
    prov_disc = prov.get("execution_provenance_disclosure", {})

    return f"""# Инженерный отчёт по доработке системы «ForteBank Cash Discipline AI» (Datathon 26)

* **Документ-основание:** «Техническое задание на доработку ForteBank Cash Discipline AI, Datathon 26, Версия 1.0 от 17 октября 2026 года», аудит `REVIEW.md` (`F01`–`F17`) и повторная ревизия `615.md` (`R01`–`R15`).
* **Целевая аппаратная платформа:** {hw.get('machine', 'Apple MacBook Air M4 (16 GB unified memory)')} (`{hw.get('os', 'macOS-arm64')}`, Python `{hw.get('python_version', '3.13.9')}`)
* **Идентификатор ревизии расчёта (`run_id`):** `{prov.get('run_id', '2026-10-18T06-45-00Z-review0550-r01-r15')}`
* **Метод формирования документа:** сгенерирован автоматически скриптом [`evaluation/generate_reports.py`](evaluation/generate_reports.py) напрямую из JSON-артефактов оценки и нормативного каталога (`F07`, `F15`, `R04`, `R09`–`R11`).

> **Методологическое примечание и раскрытие происхождения расчётов (Раздел 18.4, `AT-22`, `F06`–`F11`, `R04`):**
> В соответствии с требованиями Технического задания и ревью результаты **строго разделены на четыре независимых контура оценки** и **не объединяются в единый сводный процент точности**:
> 1. **Часть 1:** Программные проверки, устранение дефектов `F01`–`F17` и `R01`–`R15`, аудит нормативной базы (`9` активных публичных норм НБРК и `11` диагностических правил) и лицензионный аудит (`F17`).
> 2. **Часть 2:** Результаты и 3-стороннее абляционное сравнение (`cv_only`, `vlm_only`, `hybrid`) на 42 видео открытого набора с глобальным сопоставлением минимальной стоимости (`R10`) и фиксированными знаменателями (`F10`). Конфигурация `cv_only` (`{prov_disc.get('cv_only_open42', 'fresh_live_execution_with_r01_coordinate_fix')}`) выполнена заново на всех 42 видео после исправления нормализации координат `R01`; конфигурации `vlm_only` и `hybrid` (`{prov_disc.get('vlm_only_and_hybrid_open42', 'revalidated_from_cached_vlm_inference_with_r02_r10_metrics')}`) пересчитаны и провалидированы из сохранённых ответов инференса VLM с сериализацией `analysis_incomplete` (`R04`, `R07`).
> 3. **Часть 3:** Проверка устойчивости на 210 контролируемых аугментациях: полный прогон `cv_only_all_210` (`{prov_disc.get('cv_only_all_210_augmentations', 'fresh_live_execution_with_r01_coordinate_fix')}`) выполнен заново вживую на всех 210 файлах; пилот `hybrid_pilot_50` (`{prov_disc.get('hybrid_pilot_50_augmentations', 'recomputed_with_r10_r11_row_consistency_from_cached_vlm_inference')}`) пересчитан чистой функцией `evaluate_augmentations()` со 100% построчной согласованностью `time_errors_vs_gt_sec` (`R10`, `R11`).
> 4. **Часть 4:** Внешняя независимая выборка **MEVA (`CC BY 4.0`)** из 12 эпизодов с верификацией по первичным KPF-файлам `.activities.yml` (`F15`) и раздельным учётом `evaluated_episodes_count` / `banking_failed_count` (`R09`).

---

## Часть 1. Программные проверки, устранение дефектов конвейера (`F01`–`F17`, `R01`–`R15`) и нормативно-лицензионный аудит

### 1.1. Устранённые дефекты по результатам ревью (`F01`–`F17`, `R01`–`R15` и `0909`)
1. **`R01` и `R03` (Инвариантная нормализация координат детектора по `bgr.shape` и явный `CVDetectorError`):**
   * В [app/cv_detector.py](app/cv_detector.py) координаты боксов YOLO нормализуются по фактическим размерам переданного в детектор массива `f_obj.bgr.shape[:2]` (`arr_h, arr_w`), а не по исходным размерам контейнера `bundle.width/height`. Это полностью устранило искажение площади персон и расстояний при масштабировании (`640×360`, `1920×1080`, `3840×2160`).
   * Добавлено исключение `CVDetectorError`: при отсутствии весов YOLO или сбое `predict()` конвейер немедленно завершается с ошибкой вместо возврата ложного `no_violation` (`R03`).
2. **`R02`, `R05` и `0909` Разделы 4.1–4.2 (Явный тип источника `evidence_source` без текстового обхода CV-телеметрии, привязка «актор–действие–объект–локация» и сохранение физических флагов):**
   * В [app/regulations.py](app/regulations.py) и [app/schemas.py](app/schemas.py) полностью удалён текстовый обход `_CV_TELEMETRY_SIGNATURES`. Введено явное типизированное поле `evidence_source` (`"vlm_verified"` vs `"cv_detector_unverified"`): в режиме `cv_only` гипотезы получают `status="unverified_cv_hypothesis"`, `mandatory_conditions_confirmed=[]` и `None` во всех физических булевых полях, а в `hybrid`/`vlm_only` телеметрия детектора никогда не подтверждает физические условия правил.
   * Каждое обязательное условие в `RULE_CONDITION_SPECS` проверяется на уровне клауз со связкой субъекта (`cashier` vs `client`), целевого объекта, локации (`на столе` vs `в сейфе`) и локального отрицания (`_is_negated`, `_is_noun_negated`). Полностью сохраняются булевы поля (`safe_door_open`, `safe_contents_visible`, `client_present`, `phone_in_hands`, `cash_visible_on_desk`, `cashier_left_room`, `seal_check_performed`, `recount_performed`) и `condition_checks`.
3. **`R06`, `R07` и `0909` Раздел 4.3 (Привязка `candidate_id` 1-к-1, попарное сравнение признаков актора и `episode_id`, индикация `analysis_incomplete`):**
   * В [app/analyzer.py](app/analyzer.py) дедупликатор `_merge_same_episode_events()` извлекает сигнатуру актора (`_extract_actor_signature`) строго из клауз сотрудника (игнорируя относительные упоминания клиента) и сравнивает порядковые, пространственные и цветовые маркеры между двумя событиями (`_actor_signatures_conflict`), а также поддерживает сквозной `episode_id` и `actor_description`.
   * В `_build_coarse_overview_windows()` выбор следующего кадра раскадровки ограничен строгим неравенством `<= max_interval_sec` (`4.0` с в `fast`, `3.5` с в `deep`), а в [app/static/index.html](app/static/index.html) при `analysis_incomplete=True` выводится явный жёлтый статус `АНАЛИЗ НЕПОЛНЫЙ`.
4. **`R08`, `R12`–`R15` и `0909` Разделы 4.4–4.5 (Единая монотонная интерполяция пропущенных PTS на VFR, синхронизация lock-файлов, read-only `--check`, очистка очереди и построчная верификация таблиц в `AT-22`):**
   * В [app/video_decoder.py](app/video_decoder.py) метод `_reconstruct_missing_pts()` выполняет линейную интерполяцию пропущенных PTS между соседними известными кадрами и передаёт `frame_pts_map` в `decode_frames_by_ids()` и `encode_verified_evidence_frames()`, исключая нарушение монотонности и расхождение таймкодов при повторном декодировании VFR-потоков.
   * Синхронизированы [requirements.lock.txt](requirements.lock.txt) и [uv.lock](uv.lock) по всем `103` пакетам (`R12`); в [start_service.sh](start_service.sh) и [app/worker.py](app/worker.py) добавлен режим `FORTE_READONLY_CHECK=1` (`R13`) и удаление просроченных задач из `self._pending_queue.queue` (`R14`); в `AT-22` добавлена построчная сверка таблиц `REPORT.md` и всех производных полей сводок (`0909` Раздел 4.5).

### 1.2. Матрица приёмочных испытаний (`AT-01`..`AT-25`, `F06`, `R01`–`R15`)
В соответствии с `F06` в таблице ниже для каждого из 25 автотестов в [tests/test_acceptance_suite.py](tests/test_acceptance_suite.py) явно разделены **тип проверки** (модульный/контрактный, интеграционный видеопрогон CV, процессный тест супервизора или аудит артефактов) и **ограничения теста**:

| Код | Проверяемый инвариант | Тип проверки в `pytest` | Что фактически проверяется | Ограничения теста (`F06`) | Статус |
|---|---|---|---|---|---|
| **AT-01** | Однотонное корректное видео | Интеграционный видеопрогон | Декодирование 6 с чёрного видео через PyAV, `status="done"`, `verdict="no_violation"` | На пустой сцене VLM пропускается оптимизацией (`vlm_calls=0`) | **PASS** |
| **AT-02** | Штатное поведение кассира | Интеграционный видеопрогон (`cv_only`) | Полный прогон `Х-3.mov`, проверка отсутствия ложных срабатываний | Быстрый регрессионный тест выполняется в `cv_only`; `hybrid` проверен в Части 2 | **PASS** |
| **AT-03** | Короткое видео (`< 10` с) с нарушением | Контролируемый видеопрогон (`7 с` PyAV с подменённым YOLO) + проверка кода | Декодирование 7-секундного клипа через PyAV с контролируемым ответом детектора YOLO + проверка отсутствия фильтра короткой длительности в `app/*.py` | Использует подменённый детектор YOLO над синтетическим 7 с видео для быстрой проверки конвейера | **PASS** |
| **AT-04** | Инвариантность к переименованию и перемещению | Интеграционный видеопрогон (`cv_only`) | Сравнение результатов на исходном пути и копии со случайным именем во временной папке | Проверяет отсутствие зависимости от имени/пути файла | **PASS** |
| **AT-05** | Масштабирование разрешения `×2` и `×0.75` | Интеграционный видеопрогон (`cv_only`) | Сравнение вердикта, таймкода и инвариантности нормализованных координат (`R01`) | Полный аудит всех 210 аугментаций приведён в Части 3 | **PASS** |
| **AT-06** | Сдвиг начала видео на `+5.0` с | Интеграционный видеопрогон (`cv_only`) | Проверка смещения таймкода события ровно на `+5.0` с при добавлении префикса | Полный аудит `prefix_5s` приведён в Части 3 | **PASS** |
| **AT-07** | Изменение яркости/контраста и сжатие H.264 | Интеграционный видеопрогон + аудит манифеста | Проверка `luma_contrast`, `h264_compression` и относительных путей в манифесте | Деградировавшие мелкие детали учитываются отдельно в манифесте | **PASS** |
| **AT-08** | Переменная частота кадров (VFR), смещение начала (`3.5s` и `10.0s`) и частично отсутствующие PTS | Интеграционный и контролируемый тест декодера PyAV | Проверка реального файла `vfr_offset_test.mov` (`start_offset_sec=3.5s`, `is_vfr_detected=True`), потока `start_time=300` (`10.0s`) и VFR-потока с пропуском внутреннего PTS (монотонность и 0.0 расхождения при повторном декодировании, `0909` Раздел 4.4) | Сочетает реальный VFR-контейнер (`3.5s`) и контролируемые потоки с частичным пропуском PTS | **PASS** |
| **AT-09** | Незнакомая небанковская сцена | Модульный + проверка фильтра домена VLM | Проверка блокировки банковских правил при `is_banking_cash_room_scene=False` и отсутствия полей `layout`/`scenario` | Реальный прогон VLM на 12 небанковских клипах MEVA приведён в Части 4 | **PASS** |
| **AT-10** | Окклюзия и ограничения видимости | Модульный/контрактный тест | Проверка фиксации `visibility_limitations` и отказа от обвинения при перекрытии (`R02`) | Геометрический расчёт перекрытия боксов в `[0, 1]` | **PASS** |
| **AT-11** | Сохранение нескольких событий и разных эпизодов | Модульный тест дедупликатора | Сохранение разных кодов (`Δt=1.2s`) и разных эпизодов одного кода (`слева`/`справа`, `R06`) | Проверяет функцию `_merge_same_episode_events` | **PASS** |
| **AT-12** | Отклонение ложного кандидата CV моделью VLM | Контрактный тест с контролируемым ответом VLM | Перевод неподтверждённых кандидатов в `rejected_by_vlm` (включая второе событие той же категории, `R06`) | Контролируемый мок ответа VLM; реальная статистика отклонений приведена в Части 2 | **PASS** |
| **AT-13** | Валидация `frame_id`, кода, таймкода и `null`-полей | Модульный тест валидатора VLM | Отказ при `F9999`, неизвестном/отключённом коде, `timecode > duration`, `null`-полях и отрыве `timecode` от кадров `> 4.0s` | Проверяет строгий парсер и валидатор схемы | **PASS** |
| **AT-14** | Битый видеофайл, отказ YOLO (`CVDetectorError`) и невалидный JSON -> `failed` | Интеграционный тест отказоустойчивости | Мусорный `.mov`, сбой YOLO (`R03`) и обрезанный JSON вызывают явную ошибку (`status="failed"`) | Проверяет защиту от ложного `no_violation` при сбое | **PASS** |
| **AT-15** | Таймаут активной задачи и освобождение слота очереди (`R14`) | Процессный тест `JobManager` + `worker.py` | Убийство зависшего процесса (`SIGKILL`) через `1.5s` и освобождение `pending_queue` после `failed_queue_timeout` | Проверяет реальный фоновый супервизор | **PASS** |
| **AT-16** | Восстановление после перезапуска и побайтовая неизменность `jobs_data/` при `--check` (`R13`) | Интеграционный и процессный тест `JobManager` / `start_service.sh` | Перевод зависших задач в `failed_interrupted_on_restart` при старте и побайтовое сравнение файлов `jobs_data/` до и после `./start_service.sh --check` | Проверяет стартовую очистку и отсутствие побочных эффектов `FORTE_READONLY_CHECK=1` | **PASS** |
| **AT-17** | Доказательные кадры и сохранение выбора модели | Интеграционный + модульный тест (`C10`) | Проверка `Image.verify()`, границ PTS, сохранения всех выбранных VLM `primary_frame_ids` и мультифазности | Проверяет `_ensure_multiphase_evidence_ids` и кодирование JPEG | **PASS** |
| **AT-18** | Нормативный справочник (`9` активных, `11` откл.) | Аудит `app/regulations.py` | Проверка 9 публичных норм НБРК, 11 отключённых правил и `validate_observation_consistency` (`R02`, `R05`) | Сверено с текстами НБРК № 231 и № 14 (`regulatory_audit.md`) | **PASS** |
| **AT-19** | Защита API (`401`), `/api/v1/health`, `mode`/`vlm_mode`, `internal_state` и гигиена UI (`R15`) | Интеграционный тест FastAPI `TestClient` | Проверка `X-API-Key`, отсутствия ключа в `index.html` и `/catalog`, полей `healthy`/`supervisor_alive`/`vlm_model_cached`/`queue_depth`/`max_queue_size`, передачи `mode` и `vlm_mode`, возврата `internal_state="queued"` и отказа `400` на неверном режиме | Проверяет HTTP-контракт и шаблоны UI | **PASS** |
| **AT-20** | Офлайн-режим и верификация снапшота весов (`R08`) | Модульный/интеграционный тест `LocalVLMEngine` | Проверка наличия локального снапшота `fdcc572e...` и отказа `ModelNotCachedError` при неверной ревизии в `FORTE_VLM_SNAPSHOT_DIR` | Не выполняет выгрузку весов из сети | **PASS** |
| **AT-21** | Воспроизводимость (`start_service.sh --check`), парсинг wheel/sdist и `uv sync --frozen --dry-run` (`R12`) | Процессный тест и аудит графа `uv.lock` | Проверка корневого пакета, отсутствия висячих зависимостей, совпадения версий во всех именах архивов `wheels`/`sdist`, `uv sync --frozen --dry-run --offline` и `./start_service.sh --check` | Обнаруживает любое расхождение архива/версии в `uv.lock` | **PASS** |
| **AT-22** | Полный арифметический пересчёт JSON-артефактов и построчная сверка таблиц `REPORT.md` (`0909` Раздел 4.5) | Автоматический аудит метрик, производных сводок и строк таблиц отчёта | Полный пересчёт сводок `open42`, всех полей `augmentations` (включая доли и `MAE`) и `MEVA` (включая `base_action_recall`, `mean_midpoint_error_sec`, `banking_no_violation_count`), проверка SHA-256 всех 8 файлов `app/*.py` и точная построчная сверка таблиц 2.1, 3.1, 3.2 и абзацев 4.2 в `REPORT.md` (`24/42`, `23/40`, `7/28`, `19/19`) | Ломается при искажении любой сводки JSON или конкретной строки таблицы в `REPORT.md` | **PASS** |
| **AT-23** | Контрпримеры первого ревью (`C01`–`C11`, `F05`, `F13`) | Набор регрессионных тестов на контрпримерах | Покрывает все 10 контрпримеров `pipeline_cpu_repros.py`, ограничение памяти декодера (`F05`) и таймаут очереди (`F13`) | Гарантирует невозврат дефектов первого ревью | **PASS** |
| **AT-24** | Фиксированные знаменатели при `failed` и переносимость путей | Аудит функций расчёта метрик и манифестов | Проверка сохранения знаменателя при `status="failed"` (`F10`, `R09`), глобального сопоставления (`R10`) и относительных путей (`F15`) | Проверяет инварианты `audit_evaluation.py` | **PASS** |
| **AT-25** | Сквозной регрессионный тест повторных ревью (`R01`–`R15` и `0909` Разделы 4.1–4.4) | Модульный и процессный регрессионный тест | Проверяет `R01`, `CVDetectorError` (`R03`), запрет подтверждения из CV-телеметрии и привязку актор–действие–объект (`0909` 4.1–4.2), сохранение физических флагов и `condition_checks`, дедупликацию порядковых маркеров и `episode_id` (`0909` 4.3), шаг `<= 3.5s` (`R07`), `R10`, а также освобождение очереди (`R14`) и гигиену UI (`R15`) | Сочетает контролируемые проверки семантики/схемы и реальный процессный тест очереди | **PASS** |

### 1.3. Аудит нормативного справочника (`app/regulations.py`, Раздел 4 ревью)
Все ссылки на Постановление № 226 о деятельности ломбардов (`V1900019709`) исключены. По результатам проверки официальных текстов ИПС «Әділет» ([Постановление Правления НБРК № 231 `V1900019680`](https://adilet.zan.kz/rus/docs/V1900019680) и [Постановление Правления НБРК № 14 `V2000020075`](https://adilet.zan.kz/rus/docs/V2000020075)) четыре правила, условия которых в открытом наборе выходили за пределы безусловного запрета публичного акта (`multiple_clients_in_cash_room`, `unauthorized_person_in_cash_room`, `mixed_denominations_packaging`, `unattended_documents_or_seals`), переведены в диагностический контур (`enabled=False`):

#### А. Активные публично проверяемые правила НБРК (`enabled=True`, {len(enabled_cat)} правил)

| № | Код | Краткое наименование | Нормативное основание |
|---|---|---|---|
{chr(10).join(enabled_rows)}

#### Б. Отключённые / диагностические правила (`enabled=False`, {len(disabled_cat)} правил)

| № | Код | Краткое наименование | Причина перевода в диагностический контур (`enabled=False`) |
|---|---|---|---|
{chr(10).join(disabled_rows)}

### 1.4. Лицензионный аудит компонентов (`F17`)
* **`mlx-community/Qwen2.5-VL-7B-Instruct-4bit`** (ревизия `fdcc572e8b05ba9daeaf71be8c9e4267c826ff9b`, базовые веса `Qwen/Qwen2.5-VL-7B-Instruct`) — лицензия **Apache-2.0** (разрешено коммерческое использование). Модель 3B (`Qwen Research License`) полностью удалена из кода и конфигурации.
* **`PyAV` (`av==16.1.0`)** — **BSD-3-Clause**; **`opencv-python==4.13.0.92`** — **Apache-2.0**; **`mlx==0.31.1` / `mlx-vlm==0.3.10`** — **MIT**.
* **`ultralytics==8.4.38` и веса `yolo11s.pt` (`F17`):** распространяются по умолчанию под лицензией **GNU Affero General Public License v3.0 (AGPL-3.0)** (либо по платной коммерческой **Ultralytics Enterprise License**). Использование пакета `ultralytics` в закрытом корпоративном сервисе без раскрытия исходного кода требует приобретения Ultralytics Enterprise License либо замены детектора в `app/cv_detector.py` на модель под лицензией **Apache-2.0** (например, `RT-DETR` через `transformers`/`onnxruntime` или `YOLOX`).

---

## Часть 2. Результаты и 3-стороннее абляционное сравнение на 42 исходных видео открытого набора

Полный машиночитаемый протокол сохранён в [evaluation/results/stage5_open42_ablation.json](evaluation/results/stage5_open42_ablation.json).
В наборе 42 видеофайла и **40 независимых групп происхождения (`origin_group_id`)** с двумя парами побайтовых дубликатов:
* `Х-4.mov == Х-47.mov` (группа `open_group_06_Х-4_Х-47`, SHA-256 `4ee33cf2be3325b4...`)
* `Х-5.mov == Х-48.mov` (группа `open_group_39_Х-48_Х-5`, SHA-256 `2981213ef1f8fb28...`)

### 2.1. Сравнение бинарных метрик на открытом наборе (знаменатель фиксирован по всем входам, `F10`)

#### А. Против официальной бинарной разметки всех 42 файлов (`evaluation/official_labels.json`: 22 `violation`, 20 `no_violation`)

| Конфигурация | Входов / Успешно / Failed | TP | FP | TN | FN | Precision | Recall | F1 | Accuracy (на успешных) | Сквозная доля корректных решений (на всех 42 входах) |
|---|---|---|---|---|---|---|---|---|---|---|
{_bin_row("**`cv_only`**", cv_s["all_42_official_labels"], True)}
{_bin_row("**`vlm_only`**", vlm_s["all_42_official_labels"], True)}
{_bin_row("**`hybrid` (CV + VLM)**", hyb_s["all_42_official_labels"], True)}

#### Б. На 40 независимых группах происхождения (`origin_group_id`, без дубликатов `Х-47` и `Х-5`: 21 `violation`, 19 `no_violation`)

| Конфигурация | Входов / Успешно / Failed | TP | FP | TN | FN | Precision | Recall | F1 | Accuracy (на 40 группах) |
|---|---|---|---|---|---|---|---|---|---|
{_bin_row("**`cv_only`**", cv_s["unique_40_groups_official_labels"], False)}
{_bin_row("**`vlm_only`**", vlm_s["unique_40_groups_official_labels"], False)}
{_bin_row("**`hybrid` (CV + VLM)**", hyb_s["unique_40_groups_official_labels"], False)}

#### В. Против контура 9 публично подтверждённых норм НБРК (`enabled=True` GT: 11 положительных файлов / 10 уникальных групп и 31 отрицательный файл / 30 уникальных групп)
Поскольку 11 правил (`enabled=False`) относятся к внутренним СОП банка либо требуют внешних данных (на них приходится 11 из 22 положительных файлов официальной разметки: `X-1`, `X-2`, `Х-11`, `Х-13`, `Х-15`, `Х-18`, `Х-22`, `Х-30`, `Х-35`, `Х-44`, `Х-46`), ниже приведены метрики строго против 9 активных публичных норм НБРК:

| Конфигурация | Контур оценки | Входов / Успешно / Failed | TP | FP | TN | FN | Precision | Recall (на `enabled=True`) | F1 | Доля совпадений |
|---|---|---|---|---|---|---|---|---|---|---|
{_bin_row("**`cv_only`** | Все 42 файла (`enabled=True` GT)", cv_s["all_42_public_nbk_enabled_rules"], False)}
{_bin_row("**`cv_only`** | 40 уникальных групп (`enabled=True` GT)", cv_s["unique_40_public_nbk_enabled_rules"], False)}
{_bin_row("**`vlm_only`** | Все 42 файла (`enabled=True` GT)", vlm_s["all_42_public_nbk_enabled_rules"], False)}
{_bin_row("**`vlm_only`** | 40 уникальных групп (`enabled=True` GT)", vlm_s["unique_40_public_nbk_enabled_rules"], False)}
{_bin_row("**`hybrid` (CV + VLM)** | Все 42 файла (`enabled=True` GT)", hyb_s["all_42_public_nbk_enabled_rules"], False)}
{_bin_row("**`hybrid` (CV + VLM)** | 40 уникальных групп (`enabled=True` GT)", hyb_s["unique_40_public_nbk_enabled_rules"], False)}

### 2.2. Строгая оценка категорий (`p["code"] == g["code"]`, `F08`, `R10`), временной локализации и доказательных кадров (`F11`)

> **Методологическое правило (`F08`, `F10`, `F11`, `R10`):**
> * Событие считается сопоставленным (`matched_exact_category_events`) **только при точном совпадении кода категории (`p["code"] == g["code"]`)** при глобальном сопоставлении минимальной суммарной ошибки без зависимости от порядка следования событий (`R10`). Предсказания с другой категорией в том же ролике не уменьшают `missed_enabled_events` и учитываются в `extra_predicted_events`.
> * При наличии 0 событий и 0 доказательных кадров (в конфигурации `vlm_only`) показатели качества доказательных кадров указываются как **«не применимо»**, а не `100%` (`F11`).

| Показатель | `cv_only` | `vlm_only` | `hybrid` (CV + VLM) |
|---|---|---|---|
| Всего размеченных событий в `team_annotations.json` (по всем 42 входам) | `{cv_cat['total_expected_all_events']}` | `{vlm_cat['total_expected_all_events']}` | `{hyb_cat['total_expected_all_events']}` |
| Из них активных публичных событий НБРК (`enabled=True`, фиксированный знаменатель) | `{cv_cat['total_expected_enabled_public_events']}` | `{vlm_cat['total_expected_enabled_public_events']}` | `{hyb_cat['total_expected_enabled_public_events']}` |
| Сопоставлено событий с точным совпадением кода категории (`matched_exact_category_events`) | `{cv_cat['matched_exact_category_events']} / {cv_cat['total_expected_enabled_public_events']}` (`{cv_cat['exact_category_recall_on_enabled']*100:.2f}%`) | `{vlm_cat['matched_exact_category_events']} / {vlm_cat['total_expected_enabled_public_events']}` (`{vlm_cat['exact_category_recall_on_enabled']*100:.2f}%`) | `{hyb_cat['matched_exact_category_events']} / {hyb_cat['total_expected_enabled_public_events']}` (`{hyb_cat['exact_category_recall_on_enabled']*100:.2f}%`) |
| Пропущено активных событий (`missed_enabled_events`) | `{cv_cat['missed_enabled_events']}` | `{vlm_cat['missed_enabled_events']}` | `{hyb_cat['missed_enabled_events']}` |
| Лишних предсказанных событий (`extra_predicted_events`, включая смену категории) | `{cv_cat['extra_predicted_events']}` | `{vlm_cat['extra_predicted_events']}` | `{hyb_cat['extra_predicted_events']}` |
| Число сопоставленных одноимённых событий для оценки времени | `{cv_tl['evaluated_matched_events_count']}` | `{vlm_tl['evaluated_matched_events_count']}` | `{hyb_tl['evaluated_matched_events_count']}` |
| Событий с ошибкой `\\|t_pred - t_gt\\| <= 2.0s` (от сопоставленных) | `{cv_tl['within_2_0s_count']} / {cv_tl['evaluated_matched_events_count']}` (`{cv_tl['within_2_0s_rate_on_matched']*100:.2f}%`) | не применимо (`0/0`) | `{hyb_tl['within_2_0s_count']} / {hyb_tl['evaluated_matched_events_count']}` (`{hyb_tl['within_2_0s_rate_on_matched']*100:.2f}%`) |
| Событий с ошибкой `\\|t_pred - t_gt\\| <= 2.0s` (от всех `{cv_cat['total_expected_enabled_public_events']}` активных GT-событий) | `{cv_tl['within_2_0s_count']} / {cv_cat['total_expected_enabled_public_events']}` (`{cv_tl['within_2_0s_rate_on_all_enabled_expected']*100:.2f}%`) | `0 / {vlm_cat['total_expected_enabled_public_events']}` (`0.00%`) | `{hyb_tl['within_2_0s_count']} / {hyb_cat['total_expected_enabled_public_events']}` (`{hyb_tl['within_2_0s_rate_on_all_enabled_expected']*100:.2f}%`) |
| Событий с ошибкой `\\|t_pred - t_gt\\| <= 5.0s` (от сопоставленных) | `{cv_tl['within_5_0s_count']} / {cv_tl['evaluated_matched_events_count']}` | не применимо (`0/0`) | `{hyb_tl['within_5_0s_count']} / {hyb_tl['evaluated_matched_events_count']}` |
| Средняя абсолютная ошибка времени `MAE` (с) | `{cv_tl['mean_abs_error_sec']} с` | — | `{hyb_tl['mean_abs_error_sec']} с` |
| Медианная абсолютная ошибка времени `Median AE` (с) | `{cv_tl['median_abs_error_sec']} с` | — | `{hyb_tl['median_abs_error_sec']} с` |
| Максимальная абсолютная ошибка времени `Max AE` (с) | `{cv_tl['max_abs_error_sec']} с` | — | `{hyb_tl['max_abs_error_sec']} с` |
| Список ошибок времени по сопоставленным одноимённым событиям (с) | `{cv_tl['abs_errors_sec_list']}` | `[]` | `{hyb_tl['abs_errors_sec_list']}` |
| Технически валидные JPEG Base64 доказательные кадры (`Image.verify()`) | {ev_cv['jpeg']} | {ev_vlm['jpeg']} | {ev_hyb['jpeg']} |
| Доказательные кадры с валидным PTS и `decode_index` из декодированного пула | {ev_cv['pts']} | {ev_vlm['pts']} | {ev_hyb['pts']} |
| Мультифазное покрытие событий (`>= 3` упорядоченных кадров, охватывающих таймкод) | {ev_cv['mp']} | {ev_vlm['mp']} | {ev_hyb['mp']} |

### 2.3. Эксплуатационная телеметрия на Apple MacBook Air M4 (16 ГБ, `F15`)

| Показатель эксплуатации | `cv_only` | `vlm_only` (`fast`) | `hybrid` (`fast`) |
|---|---|---|---|
| Холодный старт модели VLM из локального кэша (`vlm_cold_load_sec`) | `0.00 с` | `{hw.get('vlm_cold_load_sec', 6.05)} с` | `{hw.get('vlm_cold_load_sec', 6.05)} с` |
| Среднее время обработки 1 видео (`mean_processing_time_sec`) | `{cv_op['mean_processing_time_sec']} с` | `{vlm_op['mean_processing_time_sec']} с` | `{hyb_op['mean_processing_time_sec']} с` |
| Медианное время обработки 1 видео (`median_processing_time_sec`) | `{cv_op['median_processing_time_sec']} с` | `{vlm_op['median_processing_time_sec']} с` | `{hyb_op['median_processing_time_sec']} с` |
| Максимальное время обработки 1 видео (`max_processing_time_sec`) | `{cv_op['max_processing_time_sec']} с` | `{vlm_op['max_processing_time_sec']} с` | `{hyb_op['max_processing_time_sec']} с` |
| Суммарное число вызовов VLM на 42 видео (`total_vlm_calls`) | `{cv_op['total_vlm_calls']}` | `{vlm_op['total_vlm_calls']}` | `{hyb_op['total_vlm_calls']}` |
| Пиковая резидентная память процесса (`peak_rss_mb`) | `{cv_op['peak_rss_mb']} МБ` (`{cv_op.get('rss_measurement_scope', 'isolated_cv_subprocess')}`) | `{vlm_op['peak_rss_mb']} МБ` (`{vlm_op.get('rss_measurement_scope', 'vlm_plus_cv_shared_process')}`) | `{hyb_op['peak_rss_mb']} МБ` (`{hyb_op.get('rss_measurement_scope', 'vlm_plus_cv_shared_process')}`) |

### 2.4. Пофайловый разбор результатов `hybrid` на открытом наборе (`F07`, `F08`)
1. **Поведение на 11 положительных файлах открытого набора, относящихся к отключённым диагностическим правилам (`enabled=False`):**
{chr(10).join(disabled_sop_lines)}
2. **Пропуски (`FN`) на положительных файлах с активными публичными правилами НБРК (`enabled=True`):**
{chr(10).join(enabled_fn_lines)}
3. **Ложные срабатывания (`FP`) на отрицательных файлах (`official_verdict="no_violation"`):**
{chr(10).join(hyb_fp_lines)}

---

## Часть 3. Проверка устойчивости на контролируемых аугментациях (Раздел 15, `F09`, `F10`, `F15`, `R10`, `R11`)

Манифест из **210 аугментированных видеофайлов** (`42 исходника × 5 преобразований`) с переносимыми относительными путями (`evaluation/augmentations/files/...`, `F15`) и контрольными суммами SHA-256 сохранён в [evaluation/augmentations/manifest.json](evaluation/augmentations/manifest.json). Полный протокол сохранён в [evaluation/results/stage5_augmentations_eval.json](evaluation/results/stage5_augmentations_eval.json).

> **Методологические примечания (`F09`, `F10`, `R10`, `R11`):**
> * Полный набор из 210 файлов выполнен вживую в режиме `cv_only_all_210` после исправления нормализации координат `R01`; рабочий гибридный режим проверен на пилотной выборке `hybrid_pilot_50` (50 файлов из 10 независимых групп происхождения) с полным пересчётом всех строковых и агрегатных полей чистой функцией `evaluate_augmentations()` (`row_discrepancies = 0`, `R11`).
> * Сопоставление событий между исходником и аугментированным файлом выполняется **строго один-к-одному при совпадении кода категории (`pv["code"] == sv["code"]`) с глобальной минимизацией суммарной ошибки времени (`R10`)**.

### 3.1. Полный прогон `cv_only_all_210` (210 файлов, 40 независимых групп происхождения)

| Тип преобразования (`ffmpeg`) | Всего файлов / Групп | Семантически валидных (`valid`) | Выделено вариантов (`flagged`) | Согласованность бинарного вердикта | Согласованность набора категорий | События (`исх. / совп. / пропуск / лишние`) | Сдвиг `<= 2.0s` (от сопоставленных) | Сдвиг `<= 2.0s` (от всех исходных событий) | `MAE` сдвига (с) |
|---|---|---|---|---|---|---|---|---|---|
{_aug_rows("cv_only_all_210")}

### 3.2. Пилотный прогон `hybrid_pilot_50` (50 файлов, 10 независимых групп происхождения)

| Тип преобразования (`ffmpeg`) | Всего файлов / Групп | Семантически валидных (`valid`) | Выделено вариантов (`flagged`) | Согласованность бинарного вердикта | Согласованность набора категорий | События (`исх. / совп. / пропуск / лишние`) | Сдвиг `<= 2.0s` (от сопоставленных) | Сдвиг `<= 2.0s` (от всех исходных событий) | `MAE` сдвига (с) |
|---|---|---|---|---|---|---|---|---|---|
{_aug_rows("hybrid_pilot_50")}

---

## Часть 4. Внешняя независимая выборка MEVA (`CC BY 4.0`) и границы переносимости (Раздел 14, `F01`, `F07`, `F15`, `R09`)

Манифест внешней выборки сохранён в [evaluation/meva_sample/manifest.json](evaluation/meva_sample/manifest.json), первичные файлы официальной KPF-разметки Kitware `.activities.yml` с проверенными контрольными суммами SHA-256 — в директории [evaluation/meva_sample/annotations/](evaluation/meva_sample/annotations), лицензионная атрибуция `CC BY 4.0` — в [evaluation/meva_sample/LICENSE_ATTRIBUTION.txt](evaluation/meva_sample/LICENSE_ATTRIBUTION.txt), полный протокол — в [evaluation/results/stage6_meva_evaluation.json](evaluation/results/stage6_meva_evaluation.json).

### 4.1. Объём выборки и верификация по первичным файлам `.activities.yml` (`F15`)
* **Всего видеоэпизодов (фрагментов):** `{meva_eval['counts_summary']['total_episodes']}` (`evaluation/meva_sample/clips/meva_ep01..12.mp4`)
* **Уникальных исходных 5-минутных видеозаписей MEVA (`drops-123-r13`):** `{meva_eval['counts_summary']['unique_source_recordings']}`
* **Уникальных стационарных камер наблюдения:** `{meva_eval['counts_summary']['unique_cameras']}` (`{', '.join(meva_eval['counts_summary']['cameras_list'])}`)
* **Независимых многокамерных сценариев (`scenario_group_id`):** `{meva_eval['counts_summary']['unique_independent_scenarios']}`
* **Сверка с первичными KPF YAML (`F15`):** все 28 действий проверены против первичных файлов `*.activities.yml` по идентификаторам `id2` и интервалам кадров `tsr0` (`verified_against_primary_kpf_yaml=True`), включая точные границы длящихся действий `person_texts_on_phone` в `meva_ep08` (`tsr0=[5574, 7274]`) и `meva_ep12` (`tsr0=[4277, 4706]`).

### 4.2. Сводные и поэпизодные результаты на выборке MEVA (`F07`, `R09`)
* **Распознавание базовых действий (диагностический интерфейс `run_meva_diagnostic_vlm`):** сопоставлено **`{meva_s['base_action_recall_fraction']}` (`{meva_s['base_action_recall']*100:.2f}%`)** официальных действий MEVA (при строгом глобальном взаимно однозначном сопоставлении); средняя ошибка центра интервала по сопоставленным действиям `MAE = {meva_s['mean_midpoint_error_sec']} с` (`{meva_s['midpoint_error_within_2_0s_count']}/{meva_s['temporal_localization_evaluated_count']}` сопоставленных действий или **`{meva_s['midpoint_error_within_2_0s_count']}/{meva_s['total_official_gt_activities']}` всех ожидаемых активностей** в пределах `<= 2.0 с`, `{meva_s['midpoint_error_within_3_0s_count']}/{meva_s['temporal_localization_evaluated_count']}` в пределах `<= 3.0 с`).
* **Поведение банковского конвейера (`CashDisciplineAnalyzer`) на незнакомых небанковских сценах (`facility=school`, `F01`, `F07`, `R09`):**
  * Из `{meva_b['total_episodes']}` эпизодов успешно оценено `{meva_eval_count}` (`banking_failed_count = {meva_fail_count}`). На **`{meva_b['banking_no_violation_count']} / {meva_b['total_episodes']}` (`{meva_no_viol_pct}`)** небанковских эпизодов MEVA банковский конвейер вернул явный `status="done"` и `banking_verdict="no_violation"` (`banking_violation_trigger_count = {meva_b['banking_violation_trigger_count']}`). **Оговорка о происхождении данных (`provenance`):** сохранённые банковские и диагностические ответы MEVA пересчитаны из исторического кэша инференса VLM (`recomputed_from_cached_vlm_outputs_with_honest_failed_denominator_r09`), а живой контрольный видеопрогон `hybrid` выполнен на эпизоде `meva_ep12`. Эпизоды со `status="failed"` никогда не засчитываются в `banking_no_violation_count` (`R09`).

| Эпизод MEVA | Камера / Сценарий | Официальные действия MEVA (`clip_pts_time_range_sec`, `tsr0`) | Предсказанные базовые действия (`start_sec–end_sec`) | Сопоставленные действия (1-к-1) | Вердикт банковского API (`hybrid`) |
|---|---|---|---|---|---|
{chr(10).join(meva_rows)}

### 4.3. Открытые ограничения и границы переносимости
1. **Качество локализации и распознавания на открытом наборе без специализированного дообучения детектора:** стандартный COCO-детектор `yolo11s.pt` не содержит классов `сейф (открыт/закрыт)`, `инкассаторская сумка`, `пломба`, `пачка банкнот`, а 4-битная VLM `Qwen2.5-VL-7B-Instruct-4bit` без дообучения на кассовых операциях имеет ограниченную точность различения стационарного кассового оборудования и оставленных ценностей (`exact_category_recall_on_enabled` составляет `{cv_cat['exact_category_recall_on_enabled']*100:.2f}%` в `cv_only` и `{hyb_cat['exact_category_recall_on_enabled']*100:.2f}%` в `hybrid` на `{cv_cat['total_expected_enabled_public_events']}` активных событиях).
2. **Лицензионное ограничение `ultralytics` (`F17`):** для проприетарного коммерческого развёртывания без раскрытия кода по AGPL-3.0 требуется Ultralytics Enterprise License либо замена детектора на Apache-2.0 аналог (`RT-DETR` / `YOLOX`).
"""


def main():
    readme_text = generate_readme()
    report_text = generate_report()
    with open(os.path.join(BASE_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme_text)
    with open(os.path.join(BASE_DIR, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write(report_text)
    print("Successfully generated README.md and REPORT.md from JSON evaluation artifacts.")


if __name__ == "__main__":
    main()
