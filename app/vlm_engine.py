"""
Локальный мультимодальный движок VLM на базе Apple MLX (`mlx-vlm 0.7.8`) со строгой генерацией по JSON Schema
(`mlx_vlm.structured.build_json_schema_logits_processor`) и валидацией инвариантов (Разделы 4, 5.3, 6.2, 8.2 ТЗ).

Ключевые гарантии:
1. Использует исключительно локальную коммерчески допустимую модель `mlx-community/Qwen2.5-VL-7B-Instruct-4bit`
   (Apache-2.0 License, ревизия `fdcc572e8b05ba9daeaf71be8c9e4267c826ff9b`) из локального кэша без сетевых обращений.
   При отсутствии или повреждении весов выбрасывает `ModelNotCachedError` (задача переходит в `failed`).
2. Использует `build_json_schema_logits_processor` (llguidance) из `mlx_vlm/structured.py` для гарантированного
   соблюдения JSON Schema, включая допустимые `frame_id` показанных кадров и коды справочника.
3. Запрещены регулярные выражения для извлечения вердикта из битого текста и молчаливое обрезание (`clamping`)
   выдуманных таймкодов до границ ролика. При исправимой ошибке валидации допускается ровно 1 документированная
   повторная попытка (`max_retries=1`) с записью в `vlm_retry_log`.
4. В промпте гипотезы CV передаются только как непроверенные предположения, которые VLM обязана отклонить
   при отсутствии визуального подтверждения обязательных условий правила (Раздел 5.3).
"""

import json
import math
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from app.regulations import (
    PHYSICAL_FLAG_KEYS,
    VIOLATION_CATALOG,
    evaluate_rule_conditions,
    get_disabled_catalog,
    get_enabled_catalog,
    validate_observation_consistency,
)
from app.schemas import CandidateEvent, VLMRetryRecord, build_vlm_output_json_schema
from app.video_decoder import DecodedFrame

APACHE_2_MODEL_ID = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
APACHE_2_MODEL_REVISION = "fdcc572e8b05ba9daeaf71be8c9e4267c826ff9b"

MODE_POLICIES: Dict[str, Dict[str, Any]] = {
    "fast": {
        "coarse_num_frames": 9,
        "coarse_max_interval_sec": 4.0,
        "coarse_max_windows": 8,
        "coarse_tile_size": (240, 270),
        "dense_window_half_sec": 4.0,
        "dense_num_frames": 6,
        "max_refinement_windows": 6,
        "max_tokens": 300,
    },
    "deep": {
        "coarse_num_frames": 9,
        "coarse_max_interval_sec": 3.5,
        "coarse_max_windows": 10,
        "coarse_tile_size": (280, 364),
        "dense_window_half_sec": 4.5,
        "dense_num_frames": 6,
        "max_refinement_windows": 8,
        "max_tokens": 330,
    },
}


class ModelNotCachedError(RuntimeError):
    """Обязательные локальные веса модели отсутствуют или повреждены в кэше (Раздел 4.1)."""


class VLMValidationError(RuntimeError):
    """Ответ VLM не прошёл синтаксическую или смысловую валидацию после допустимой повторной попытки (Раздел 8.2)."""


class SemanticRejectionError(ValueError):
    """
    Структурно корректное событие VLM не подтверждает необходимые условия правила связкой
    «участник — действие — объект» (ревью 0909, п. 4.1–4.2). В отличие от структурной ошибки ответа,
    такое событие после повторной попытки отбрасывается (не публикуется), а не переводит задачу в failed.
    """


import re

_FOREIGN_DOMAIN_MARKERS = (
    "небанковск",
    "общественное помещение",
    "комната отдыха",
    "учебная аудитория",
    "учебный класс",
    "офисный коридор",
    "вестибюль",
    "уличная камера",
    "спортзал",
    "спортивн",
    "баскетбол",
    "трибун",
    "парковк",
    "автостоянк",
    "лестничн",
    "лестниц",
    "тротуар",
    "крыльц",
)
_FOREIGN_DOMAIN_WITHOUT_DESK_RE = re.compile(
    r"\b(?:улиц\w*|коридор\w*|холл\w*|школ\w*|двор\w*|проезж\w*|склад\w*|цех\w*|зал\w*)\b"
)
_DESK_OR_CASH_CONTEXT_RE = re.compile(
    r"\b(?:стол\w*|касс\w*|сейф\w*|банкнот\w*|купюр\w*|наличн\w*|ноутбук\w*|компьютер\w*|монитор\w*)\b"
)
_WORKSTATION_INTERIOR_RE = re.compile(
    r"\b(?:офис\w*|кабинет\w*|рабоч\w+\s+(?:стол\w*|мест\w*|помещени\w*|комнат\w*|зон\w*))\b"
)

_CANDIDATE_VERIFICATION_HINTS: Dict[str, str] = {
    "unattended_cash_on_desk_on_exit": (
        "Проверь по кадрам до и после ухода: 1) лежат ли на рабочем столе банкноты/наличные/купюры; "
        "2) встал ли кассир из-за стола и вышел ли из помещения (оставил стол без присмотра). "
        "Если оба факта подтверждаются — укажи code='unattended_cash_on_desk_on_exit' и observed_facts: "
        "«Банкноты лежат на рабочем столе; кассир вышел из помещения». Если стол пуст или кассир не выходил — отклони."
    ),
    "unattended_previous_client_cash": (
        "Проверь по кадрам смены людей у стола: 1) остались ли на рабочем столе неубранные банкноты/наличные первого клиента; "
        "2) подошёл ли к столу кассира следующий (новый) клиент до того, как эти деньги были убраны со стола. "
        "Если оба факта подтверждаются — укажи code='unattended_previous_client_cash' и observed_facts: "
        "«На рабочем столе лежат неубранные банкноты предыдущего клиента; следующий клиент подошел к столу кассира». "
        "Если деньги убраны до подхода нового клиента или смены клиента нет — отклони."
    ),
    "counting_in_client_absence": (
        "Проверь: 1) пересчитывает ли кассир банкноты/наличные за рабочим столом; "
        "2) вышел ли клиент из помещения во время пересчёта, пока кассир остался за столом. "
        "Если кассир тоже встаёт и уходит или работает только с бумагами/ноутбуком — отклони в rejected_hypothesis_codes. "
        "Если пересчёт банкнот в отсутствие клиента подтверждается — укажи observed_facts: "
        "«Кассир пересчитывает банкноты на рабочем столе; клиент вышел из помещения»."
    ),
    "personal_phone_usage": (
        "Внимание: детектор часто путает смартфон с компьютерной мышью, калькулятором, пачкой купюр или телефоном клиента. "
        "Если телефон у клиента, лежит на столе или кассир работает с деньгами/бумагами/ноутбуком — обязательно отклони в rejected_hypothesis_codes. "
        "Подтверждай ТОЛЬКО если отчётливо видно, что сам кассир за столом держит смартфон у уха или перед лицом."
    ),
    "safe_left_open_during_work": (
        "Проверь: открыта ли дверца сейфа с видимыми внутри банкнотами/ценностями, пока кассир работает за столом и не обращается к сейфу. "
        "Если да — укажи observed_facts: «Дверца сейфа открыта и внутри сейфа видны банкноты; кассир работает за столом и не обращается к сейфу»."
    ),
    "safe_left_open_on_exit": (
        "Проверь: открыта ли дверца сейфа с видимыми внутри банкнотами/ценностями после выхода кассира из помещения. "
        "Если да — укажи observed_facts: «Дверца сейфа открыта и внутри сейфа видны банкноты; кассир вышел из помещения»."
    ),
    "unattended_collection_bag_on_desk": (
        "Проверь: оставлена ли инкассаторская сумка на рабочем столе после выхода кассира из помещения. "
        "Если да — укажи observed_facts: «Инкассаторская сумка лежит на рабочем столе; кассир вышел из помещения»."
    ),
    "collection_bag_accepted_without_check": (
        "Проверь: принял ли кассир инкассаторскую сумку без осмотра пломбы и сопроводительных документов. "
        "Если да — укажи observed_facts: «Кассир принял инкассаторскую сумку; кассир не проверил пломбу и документы»."
    ),
    "personal_belongings_outerwear_on_desk": (
        "Проверь: положил ли кассир куртку/верхнюю одежду или личную сумку на рабочий стол. "
        "Если да — укажи observed_facts: «Кассир положил куртку и личную сумку на рабочий стол»."
    ),
}


class LocalVLMEngine:
    """
    Двухпроходный локальный VLM-верификатор и анализатор сцен на базе Qwen2.5-VL-7B-Instruct-4bit (MLX).
    """

    def __init__(self, default_mode: str = "auto", raw_response_log: Optional[List[Dict[str, Any]]] = None):
        self.default_mode = default_mode
        self._loaded_bundle: Optional[Tuple[Any, Any, Any, str]] = None
        # Необязательный журнал сырых ответов модели (для воспроизводимых оценочных прогонов, ревью 0909 п. 7.6)
        self.raw_response_log = raw_response_log

    def get_local_snapshot_path(self) -> Optional[str]:
        """Возвращает путь к локальной папке снапшота весов строго проверенной ревизии APACHE_2_MODEL_REVISION (R08)."""
        override_dir = os.environ.get("FORTE_VLM_SNAPSHOT_DIR")
        if override_dir is not None:
            norm_override = os.path.normpath(override_dir)
            if not os.path.isdir(norm_override):
                return None
            if os.path.basename(norm_override) != APACHE_2_MODEL_REVISION:
                return None
            st_files = [f for f in os.listdir(norm_override) if f.endswith(".safetensors")]
            if st_files and os.path.exists(os.path.join(norm_override, "config.json")):
                return norm_override
            return None

        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        folder_name = "models--" + APACHE_2_MODEL_ID.replace("/", "--")
        snapshots_root = os.path.join(cache_dir, folder_name, "snapshots")
        if not os.path.isdir(snapshots_root):
            return None

        preferred = os.path.join(snapshots_root, APACHE_2_MODEL_REVISION)
        if os.path.isdir(preferred):
            st_files = [f for f in os.listdir(preferred) if f.endswith(".safetensors")]
            if st_files and os.path.exists(os.path.join(preferred, "config.json")):
                return preferred
        return None

    def is_model_cached(self, mode: str = "deep") -> bool:
        return self.get_local_snapshot_path() is not None

    @staticmethod
    def resolve_mode(
        requested_mode: str,
        duration_sec: float,
        cv_candidates_count: int = 0,
    ) -> str:
        """
        Воспроизводимое правило выбора вычислительного режима ('fast' или 'deep') по Разделу 4.2 ТЗ.
        Длительность влияет исключительно на вычислительный бюджет (число кадров и окон уточнения),
        но никогда не определяет категорию или таймкод нарушения.
        """
        if requested_mode in ("fast", "deep"):
            return requested_mode
        if duration_sec >= 30.0 or cv_candidates_count >= 1:
            return "deep"
        return "fast"

    def _load(self) -> Tuple[Any, Any, Any, str]:
        snap_path = self.get_local_snapshot_path()
        if not snap_path:
            raise ModelNotCachedError(
                f"Локальные веса модели {APACHE_2_MODEL_ID} (ревизия {APACHE_2_MODEL_REVISION}) "
                f"не найдены в локальном кэше. Сетевая подгрузка во время запроса запрещена (Раздел 4.1)."
            )
        if self._loaded_bundle is not None and self._loaded_bundle[3] == snap_path:
            return self._loaded_bundle

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        # Загружаем напрямую из локальной директории снапшота, исключая любые сетевые вызовы
        model, processor = load(snap_path)
        config = load_config(snap_path)
        self._loaded_bundle = (model, processor, config, snap_path)
        return self._loaded_bundle

    @staticmethod
    def _classify_scene_domain(scene_desc: str, model_is_banking: bool) -> Tuple[bool, str, Optional[str]]:
        """
        Разделяет уверенно чужой домен (улица, парковка, лестница, коридор без рабочего стола, спортзал,
        учебная аудитория) и рабочее/офисное помещение со столом, где выполняются целевые операции (ревью 1017, N01).
        Возвращает `(effective_is_banking_scene, domain_status, domain_rejection_reason)`.
        """
        low_scene = (scene_desc or "").strip().lower()
        for marker in _FOREIGN_DOMAIN_MARKERS:
            if marker in low_scene:
                return (
                    False,
                    "rejected_non_banking_domain",
                    f"Сцена отнесена к внешнему небанковскому домену по признаку '{marker}': {scene_desc}",
                )
        if _FOREIGN_DOMAIN_WITHOUT_DESK_RE.search(low_scene) and not _DESK_OR_CASH_CONTEXT_RE.search(low_scene):
            return (
                False,
                "rejected_non_banking_domain",
                f"Сцена описывает коридор/улицу/холл без рабочего стола или кассовой зоны: {scene_desc}",
            )
        if model_is_banking:
            return True, "banking_or_workstation_scene", None
        if (
            _WORKSTATION_INTERIOR_RE.search(low_scene)
            and "без стола" not in low_scene
            and "нет стола" not in low_scene
            and "отсутствует стол" not in low_scene
        ):
            return True, "office_workstation_interior", None
        return (
            False,
            "rejected_non_banking_domain",
            f"Модель отклонила принадлежность сцены к рабочему/кассовому помещению: {scene_desc}",
        )

    @classmethod
    def _validate_structured_payload(
        cls,
        raw_text: str,
        allowed_frame_ids: List[str],
        duration_sec: float,
        frame_timestamps: Optional[Dict[str, float]] = None,
        semantic_rejections: str = "raise",
    ) -> Dict[str, Any]:
        """
        Строгая синтаксическая и семантическая проверка ответа VLM (Раздел 8.2, F01, F04, Section 6 P2 #3-#4):
        - полный разбор `json.loads` без regex-восстановления;
        - строгая проверка типов всех обязательных полей (запрет `null` / `None`);
        - проверка всех `code` по справочнику;
        - проверка `timecode_sec` строго в диапазоне `[0.0, duration_sec]` без `eps` и без молчаливого `clamping`;
        - проверка принадлежности `evidence_frame_ids` множеству `allowed_frame_ids` и близости их временных меток
          к `timecode_sec` (`min_diff <= 4.0s`);
        - проверка непротиворечивости `observed_facts` условиям правила.
        """
        if not raw_text or not raw_text.strip():
            raise ValueError("Пустой ответ от VLM.")

        try:
            data = json.loads(raw_text.strip())
        except Exception as e:
            raise ValueError(f"Синтаксическая ошибка JSON ответа VLM: {e}") from e

        if not isinstance(data, dict):
            raise ValueError("Ответ VLM не является JSON-объектом.")

        required_keys = {
            "scene_description",
            "visibility_limitations",
            "rejected_hypothesis_codes",
            "confirmed_events",
            "diagnostic_observations",
        }
        missing = required_keys - set(data.keys())
        if missing:
            raise ValueError(f"В JSON-ответе VLM отсутствуют обязательные поля: {sorted(missing)}")

        scene_desc_raw = data.get("scene_description")
        if not isinstance(scene_desc_raw, str) or len(scene_desc_raw.strip()) < 3:
            raise ValueError(
                f"Поле 'scene_description' должно быть непустой строкой, получено: {type(scene_desc_raw).__name__}"
            )
        scene_desc = scene_desc_raw.strip()

        for list_key in (
            "visibility_limitations",
            "rejected_hypothesis_codes",
            "confirmed_events",
            "diagnostic_observations",
        ):
            val = data.get(list_key)
            if not isinstance(val, list):
                raise ValueError(
                    f"Поле '{list_key}' должно быть списком (list), получено: {type(val).__name__}"
                )

        raw_model_is_banking = True
        if "is_banking_cash_room_scene" in data:
            if not isinstance(data["is_banking_cash_room_scene"], bool):
                raise ValueError("Поле 'is_banking_cash_room_scene' должно иметь тип boolean.")
            raw_model_is_banking = bool(data["is_banking_cash_room_scene"])

        is_banking_scene, domain_status, domain_rejection_reason = cls._classify_scene_domain(
            scene_desc=scene_desc,
            model_is_banking=raw_model_is_banking,
        )

        enabled_cat = get_enabled_catalog()
        allowed_fids_set = set(allowed_frame_ids)
        effective_ts_map: Dict[str, float] = dict(frame_timestamps or {})
        if "F0000" in allowed_fids_set and "F0000" not in effective_ts_map:
            effective_ts_map["F0000"] = 0.0

        visibility_lims = [str(x).strip() for x in data["visibility_limitations"] if isinstance(x, str) and str(x).strip()]
        rejected_codes = [str(x).strip() for x in data["rejected_hypothesis_codes"] if isinstance(x, str) and str(x).strip()]

        if not is_banking_scene and domain_rejection_reason:
            if domain_rejection_reason not in visibility_lims:
                visibility_lims.append(domain_rejection_reason)

        if semantic_rejections not in ("raise", "drop"):
            raise ValueError(f"Недопустимый режим semantic_rejections='{semantic_rejections}'.")
        validated_confirmed = []
        semantically_rejected: List[Dict[str, Any]] = []
        for idx, ev in enumerate(data["confirmed_events"]):
            if not isinstance(ev, dict):
                raise ValueError(f"confirmed_events[{idx}] не является объектом.")
            code = str(ev.get("code") or "").strip()
            if code not in VIOLATION_CATALOG:
                raise ValueError(f"Неизвестный код нарушения '{code}' в confirmed_events[{idx}].")
            if code not in enabled_cat:
                raise ValueError(
                    f"Отключённое правило '{code}' (enabled=False) возвращено в confirmed_events[{idx}]."
                )
            tc = ev.get("timecode_sec")
            if not isinstance(tc, (int, float)) or isinstance(tc, bool) or not math.isfinite(float(tc)):
                raise ValueError(f"Некорректный timecode_sec '{tc}' в confirmed_events[{idx}].")
            tc_val = float(tc)
            if tc_val < 0.0 or tc_val > duration_sec:
                raise ValueError(
                    f"Выдуманный или выходящий за границы видео timecode_sec={tc_val}s "
                    f"(допустимый диапазон [0.0, {duration_sec:.2f}s]) в confirmed_events[{idx}]."
                )
            obs_raw = ev.get("observed_facts")
            if not isinstance(obs_raw, str) or len(obs_raw.strip()) < 10:
                raise ValueError(f"Слишком короткое или некорректное описание фактов observed_facts в confirmed_events[{idx}].")
            obs = obs_raw.strip()

            fids = ev.get("evidence_frame_ids")
            if not isinstance(fids, list) or not (1 <= len(fids) <= 3):
                raise ValueError(f"evidence_frame_ids должен содержать от 1 до 3 кадров в confirmed_events[{idx}].")
            for fid in fids:
                if not isinstance(fid, str) or fid not in allowed_fids_set:
                    raise ValueError(
                        f"Несуществующий или не показанный на раскадровке кадр '{fid}' в confirmed_events[{idx}]."
                    )

            # F04: Проверка связности timecode_sec с реальным временем доказательных кадров
            known_ev_times = [float(effective_ts_map[fid]) for fid in fids if fid in effective_ts_map]
            if known_ev_times:
                min_diff = min(abs(ft - tc_val) for ft in known_ev_times)
                if min_diff > 4.0:
                    raise ValueError(
                        f"Время события confirmed_events[{idx}] (timecode_sec={tc_val:.2f}s) не соответствует "
                        f"временным меткам выбранных доказательных кадров {fids} (timestamps={known_ev_times}, "
                        f"min_diff={min_diff:.2f}s > 4.0s)."
                    )

            dedup_fids = list(dict.fromkeys(fids))

            if not is_banking_scene:
                if code not in rejected_codes:
                    rejected_codes.append(code)
                note = "Сцена не является банковским кассовым помещением; банковские нормы НБРК № 231 не применимы."
                if note not in visibility_lims:
                    visibility_lims.append(note)
                continue

            structured_conds: Dict[str, Any] = {}
            for flag_key in PHYSICAL_FLAG_KEYS + ("condition_checks",):
                if flag_key in ev and ev[flag_key] is not None:
                    structured_conds[flag_key] = ev[flag_key]

            # F01 / R02 / R05 / ревью 0909: каждое необходимое условие должно быть установлено связкой
            # «участник — действие — объект» в observed_facts; флаги не заменяют текстового подтверждения.
            try:
                eval_conds = evaluate_rule_conditions(
                    code=code,
                    observed_facts=obs,
                    structured_conditions=structured_conds,
                    evidence_frame_ids=dedup_fids,
                )
                validate_observation_consistency(
                    code=code,
                    observed_facts=obs,
                    structured_conditions=structured_conds,
                    evidence_frame_ids=dedup_fids,
                )
            except ValueError as sem_err:
                if semantic_rejections == "drop":
                    semantically_rejected.append(
                        {
                            "code": code,
                            "timecode_sec": round(tc_val, 2),
                            "observed_facts": obs,
                            "evidence_frame_ids": dedup_fids,
                            "reason": str(sem_err),
                        }
                    )
                    continue
                raise SemanticRejectionError(f"confirmed_events[{idx}]: {sem_err}") from sem_err

            parsed_item: Dict[str, Any] = {
                "code": code,
                "timecode_sec": round(tc_val, 2),
                "observed_facts": obs,
                "evidence_frame_ids": dedup_fids,
                "mandatory_conditions_confirmed": eval_conds["confirmed_conditions"],
                "condition_checks": eval_conds["condition_checks"],
            }
            # Сохраняются только проверенные флаги (связанные с условиями правила); несвязанные явные флаги
            # модели не переносятся в результат.
            for k_flag, v_flag in eval_conds["physical_flags"].items():
                if v_flag is not None:
                    parsed_item[k_flag] = v_flag

            raw_cid = ev.get("candidate_id")
            if isinstance(raw_cid, str) and raw_cid.strip():
                parsed_item["candidate_id"] = raw_cid.strip()
            raw_epid = ev.get("episode_id")
            if isinstance(raw_epid, str) and raw_epid.strip():
                parsed_item["episode_id"] = raw_epid.strip()
            raw_actor_desc = ev.get("actor_description")
            if isinstance(raw_actor_desc, str) and raw_actor_desc.strip():
                parsed_item["actor_description"] = raw_actor_desc.strip()

            validated_confirmed.append(parsed_item)

        validated_diag = []
        for idx, dob in enumerate(data["diagnostic_observations"]):
            if not isinstance(dob, dict):
                raise ValueError(f"diagnostic_observations[{idx}] не является объектом.")
            code = str(dob.get("code") or "").strip()
            if code not in VIOLATION_CATALOG:
                raise ValueError(f"Неизвестный код '{code}' в diagnostic_observations[{idx}].")
            tc = dob.get("timecode_sec")
            if not isinstance(tc, (int, float)) or isinstance(tc, bool) or not math.isfinite(float(tc)):
                raise ValueError(f"Некорректный timecode_sec в diagnostic_observations[{idx}].")
            tc_val = float(tc)
            if tc_val < 0.0 or tc_val > duration_sec:
                raise ValueError(
                    f"timecode_sec={tc_val}s вне диапазона [0.0, {duration_sec:.2f}s] в diagnostic_observations[{idx}]."
                )
            fids = dob.get("referenced_frame_ids")
            if not isinstance(fids, list) or not fids:
                raise ValueError(f"referenced_frame_ids должен быть непустым списком в diagnostic_observations[{idx}].")
            for fid in fids:
                if not isinstance(fid, str) or fid not in allowed_fids_set:
                    raise ValueError(f"Неизвестный frame_id '{fid}' в diagnostic_observations[{idx}].")
            known_diag_times = [float(effective_ts_map[fid]) for fid in fids if fid in effective_ts_map]
            if known_diag_times:
                min_diff = min(abs(ft - tc_val) for ft in known_diag_times)
                if min_diff > 4.0:
                    raise ValueError(
                        f"Время diagnostic_observations[{idx}] ({tc_val:.2f}s) не связано с кадрами {fids} ({known_diag_times})."
                    )
            validated_diag.append(
                {
                    "code": code,
                    "timecode_sec": round(tc_val, 2),
                    "observed_facts": str(dob.get("observed_facts") or "").strip(),
                    "referenced_frame_ids": list(dict.fromkeys(fids)),
                }
            )

        return {
            "is_banking_cash_room_scene": is_banking_scene,
            "raw_model_is_banking_scene": raw_model_is_banking,
            "domain_status": domain_status,
            "domain_rejection_reason": domain_rejection_reason,
            "scene_description": scene_desc,
            "visibility_limitations": visibility_lims,
            "rejected_hypothesis_codes": rejected_codes,
            "confirmed_events": validated_confirmed,
            "diagnostic_observations": validated_diag,
            "semantically_rejected_events": semantically_rejected,
        }

    def run_vlm_pass(
        self,
        stage_name: str,
        storyboard_img: Image.Image,
        shown_frames: List[DecodedFrame],
        duration_sec: float,
        cv_hypotheses: Optional[List[CandidateEvent]] = None,
        max_tokens: int = 450,
    ) -> Tuple[Dict[str, Any], List[VLMRetryRecord], int]:
        """
        Выполняет один проход VLM (обзорный или плотный уточняющий) со строгим ограничением генерации
        по JSON Schema (`build_json_schema_logits_processor`) и максимум 1 повторной попыткой при ошибке валидации.
        Возвращает `(validated_payload, retry_records, calls_made)`.
        """
        model, processor, config, _ = self._load()
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.structured import build_json_schema_logits_processor

        allowed_fids = [f.frame_id for f in shown_frames]
        frame_ts_map = {f.frame_id: float(f.timestamp_sec) for f in shown_frames}
        frame_time_table = ", ".join(f"{f.frame_id}={f.timestamp_sec:.2f}s" for f in shown_frames)

        enabled_rules_text = "\n".join(
            f"- {code}: {meta['short_title']} ({meta['mandatory_visual_indicators'][0]})"
            for code, meta in sorted(get_enabled_catalog().items())
        )
        disabled_rules_text = ", ".join(sorted(get_disabled_catalog().keys()))

        if cv_hypotheses:
            hyp_lines = []
            for h in cv_hypotheses:
                closest_shown = sorted(shown_frames, key=lambda f: abs(f.timestamp_sec - h.timecode_sec))[:3]
                closest_shown_ordered = sorted(closest_shown, key=lambda f: f.timestamp_sec)
                shown_ref_str = ", ".join(f"{f.frame_id}={f.timestamp_sec:.2f}s" for f in closest_shown_ordered)
                hint_str = _CANDIDATE_VERIFICATION_HINTS.get(h.code, h.observed_facts)
                hyp_lines.append(
                    f"* Кандидат {h.candidate_id} (код=`{h.code}`, t≈{h.timecode_sec:.2f}s, кадры [{shown_ref_str}]): {hint_str}"
                )
            hyp_block = (
                "Предварительные кандидаты событий для визуальной проверки по кадрам раскадровки:\n"
                + "\n".join(hyp_lines)
                + "\nЕсли признаки кандидата видны на кадрах — обязательно включи его в confirmed_events с точным timecode_sec и кадрами из раскадровки. "
                "Если обязательные условия НЕ выполняются (стол пуст, сейф закрыт, штатная работа) — укажи код в rejected_hypothesis_codes."
            )
        else:
            hyp_block = "Выполни самостоятельный визуальный осмотр всех кадров раскадровки."

        base_prompt = (
            f"Видеоанализ кассовой дисциплины ({stage_name}, длительность {duration_sec:.2f}с).\n"
            f"Кадры раскадровки (ID=время): [{frame_time_table}].\n"
            f"{hyp_block}\n\n"
            f"Разрешённые коды нарушений НБРК (для confirmed_events):\n{enabled_rules_text}\n"
            f"Коды внутренних СОП (ТОЛЬКО для diagnostic_observations): [{disabled_rules_text}].\n\n"
            "Правила проверки:\n"
            "1. Сначала кратко опиши сцену в scene_description и установи is_banking_cash_room_scene=true для любого кассового, операционного или рабочего/офисного помещения со столом (даже в обычном кабинете без бронестекла). Ставь false ТОЛЬКО для улицы, парковки, лестницы, коридора без стола, спортзала или учебной аудитории.\n"
            "2. Проверь по кадрам: оставлены ли банкноты на рабочем столе после выхода кассира из помещения (unattended_cash_on_desk_on_exit); "
            "подошёл ли следующий клиент к столу при неубранных банкнотах предыдущего клиента (unattended_previous_client_cash); "
            "продолжает ли кассир пересчёт банкнот на столе после ухода клиента (counting_in_client_absence); "
            "держит ли кассир смартфон в руке (personal_phone_usage); "
            "открыта ли дверца сейфа с банкнотами без обращения к сейфу (safe_left_open_during_work / safe_left_open_on_exit); "
            "оставлена ли на столе инкассаторская сумка при выходе кассира (unattended_collection_bag_on_desk); "
            "принята ли инкассаторская сумка без проверки пломбы и документов (collection_bag_accepted_without_check); "
            "положены ли куртка или личная сумка кассира на рабочий стол (personal_belongings_outerwear_on_desk).\n"
            f"3. В confirmed_events указывай timecode_sec выбранного кадра из [{frame_time_table}], 1–3 кадра из [{', '.join(allowed_fids)}] и точные факты через точку с запятой (КТО — ЧТО делает — С КАКИМ объектом — ГДЕ)."
        )

        schema_dict = build_vlm_output_json_schema(allowed_fids)
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            storyboard_img.save(tmp_path, quality=85)

        retry_log: List[VLMRetryRecord] = []
        calls_made = 0
        first_error_msg = ""
        retry_kind = "validation"

        try:
            import mlx.core as mx
        except Exception:
            mx = None

        try:
            for attempt in (1, 2):
                prompt_text = base_prompt
                if attempt == 2:
                    if retry_kind == "domain_recheck":
                        prompt_text += (
                            f"\nВНИМАНИЕ (повторная проверка рабочего помещения: {first_error_msg}): "
                            "Офисное помещение или кабинет с рабочим столом ЯВЛЯЕТСЯ допустимой сценой кассовой дисциплины "
                            "(обязательно установи is_banking_cash_room_scene=true). Внимательно проверь кандидатов на раскадровке: "
                            "если признаки нарушения видны на кадрах, включи событие в confirmed_events с точным timecode_sec из "
                            f"[{frame_time_table}] и полным описанием 'КТО — ЧТО делает — С КАКИМ объектом — ГДЕ'."
                        )
                    else:
                        prompt_text += (
                            f"\nВНИМАНИЕ (повтор после ошибки: {first_error_msg}): "
                            f"timecode_sec должен совпадать со временем выбранных кадров из [{frame_time_table}] в [0.0, {duration_sec:.2f}]; "
                            "каждое событие должно явно называть участника, действие, объект и место каждого обязательного условия."
                        )

                formatted_prompt = apply_chat_template(
                    processor, config, prompt_text, num_images=1
                )
                logits_processor = build_json_schema_logits_processor(tokenizer, schema_dict)
                calls_made += 1
                result = generate(
                    model,
                    processor,
                    formatted_prompt,
                    image=[tmp_path],
                    logits_processors=[logits_processor],
                    max_tokens=max_tokens,
                    temp=0.0,
                    verbose=False,
                )
                if mx is not None and hasattr(mx, "clear_cache"):
                    mx.clear_cache()
                raw_text = result.text if hasattr(result, "text") else str(result)
                if self.raw_response_log is not None:
                    self.raw_response_log.append(
                        {
                            "stage": stage_name,
                            "attempt": attempt,
                            "shown_frame_ids": list(allowed_fids),
                            "cv_hypothesis_ids": [h.candidate_id for h in (cv_hypotheses or [])],
                            "raw_text": raw_text,
                        }
                    )

                try:
                    validated = self._validate_structured_payload(
                        raw_text=raw_text,
                        allowed_frame_ids=allowed_fids,
                        duration_sec=duration_sec,
                        frame_timestamps=frame_ts_map,
                    )
                    if (
                        attempt == 1
                        and not validated.get("raw_model_is_banking_scene", True)
                        and validated.get("domain_status") == "office_workstation_interior"
                        and not validated.get("confirmed_events")
                    ):
                        first_error_msg = (
                            f"Сцена '{validated.get('scene_description')}' ошибочно помечена is_banking_cash_room_scene=false "
                            "при наличии рабочего помещения со столом"
                        )
                        retry_kind = "domain_recheck"
                        continue
                    if attempt == 2:
                        retry_log.append(
                            VLMRetryRecord(
                                stage=stage_name,
                                attempt=1,
                                initial_error=first_error_msg,
                                retry_outcome=(
                                    "recovered_domain_rechecked"
                                    if retry_kind == "domain_recheck"
                                    else "recovered"
                                ),
                            )
                        )
                    return validated, retry_log, calls_made
                except SemanticRejectionError as sem_err:
                    if attempt == 1:
                        first_error_msg = str(sem_err)
                        retry_kind = "validation"
                        continue
                    # Структура ответа корректна, но часть событий не подтверждает условия правила:
                    # такие события не публикуются (отклоняются), остальной ответ используется.
                    try:
                        validated = self._validate_structured_payload(
                            raw_text=raw_text,
                            allowed_frame_ids=allowed_fids,
                            duration_sec=duration_sec,
                            frame_timestamps=frame_ts_map,
                            semantic_rejections="drop",
                        )
                    except Exception as struct_err:
                        retry_log.append(
                            VLMRetryRecord(
                                stage=stage_name,
                                attempt=1,
                                initial_error=first_error_msg,
                                retry_outcome="failed",
                            )
                        )
                        raise VLMValidationError(
                            f"VLM не прошла валидацию после повторной попытки на этапе '{stage_name}': {struct_err}"
                        ) from struct_err
                    retry_log.append(
                        VLMRetryRecord(
                            stage=stage_name,
                            attempt=1,
                            initial_error=f"{first_error_msg} | повтор: {sem_err}",
                            retry_outcome="recovered_semantic_events_rejected",
                        )
                    )
                    return validated, retry_log, calls_made
                except Exception as val_err:
                    if attempt == 1:
                        first_error_msg = str(val_err)
                        retry_kind = "validation"
                        continue
                    retry_log.append(
                        VLMRetryRecord(
                            stage=stage_name,
                            attempt=1,
                            initial_error=first_error_msg,
                            retry_outcome="failed",
                        )
                    )
                    raise VLMValidationError(
                        f"VLM не прошла валидацию после повторной попытки на этапе '{stage_name}': {val_err}"
                    ) from val_err

            raise VLMValidationError(f"Не удалось получить валидный ответ VLM на этапе '{stage_name}'.")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

