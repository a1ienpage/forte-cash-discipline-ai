"""
Строгие модели данных и межполевая валидация на базе Pydantic v2 (Раздел 8 ТЗ на доработку).

Обеспечивает:
1. Внутренние и выходные схемы для:
   - наблюдений (`ObservationRecord`),
   - кандидатов событий (`CandidateEvent`),
   - доказательных кадров (`EvidenceFrameModel`),
   - нормативного основания (`RegulationBasisModel`),
   - подтверждённого нарушения (`VerifiedViolationModel`),
   - диагностики и отклонённых гипотез (`AnalysisDiagnosticsModel`),
   - итогового ответа анализатора (`AnalysisResultModel`),
   - состояния задачи воркера (`JobStateModel`).
2. Проверку всех обязательных инвариантов Раздела 8.2:
   - `violation` требует непустого `violations`; `no_violation` требует пустого `violations`;
   - `code` обязан существовать в `VIOLATION_CATALOG` и иметь `enabled=True` (неизвестный или отключённый код
     вызывает явную ошибку валидации и никогда не преобразуется в другую категорию или `no_violation`);
   - `timecode_sec` и `timestamp_sec` должны быть конечными числами (`math.isfinite`) в диапазоне `[0, duration_sec]`
     без молчаливого обрезания (`clamping`) до границ видео;
   - все `frame_id` доказательств обязаны принадлежать множеству реально декодированных кадров этапа анализа;
   - `image_base64` должен декодироваться в валидный JPEG.
3. JSON Schema для ограниченной генерации `mlx_vlm.structured.build_json_schema_logits_processor` (MLX-VLM 0.7.8).
"""

import base64
import io
import math
from typing import Any, Dict, List, Literal, Optional, Set

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from app.regulations import (
    EVIDENCE_SOURCE_CV,
    EVIDENCE_SOURCE_VLM,
    PHYSICAL_FLAG_KEYS,
    VIOLATION_CATALOG,
    evaluate_rule_conditions,
    get_enabled_catalog,
    validate_observation_consistency,
)


class ObservationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_id: str
    pts: int
    timestamp_sec: float
    person_count: int = Field(ge=0)
    objects: List[str] = Field(default_factory=list)
    motion_score: float = Field(ge=0.0)
    mean_luma: float = Field(ge=0.0)
    visibility_notes: List[str] = Field(default_factory=list)


class CandidateEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    code: str
    observed_facts: str = Field(min_length=5)
    referenced_frame_ids: List[str] = Field(min_length=1)
    estimated_start_sec: float
    estimated_end_sec: float
    timecode_sec: float
    conditions_to_verify: List[str] = Field(default_factory=list)
    origin: Literal["cv_hypothesis", "vlm_coarse", "vlm_refined", "combined_cv_vlm"]
    verification_status: Literal[
        "pending",
        "confirmed_violation",
        "diagnostic_only_disabled_rule",
        "rejected_by_vlm",
        "rejected_missing_conditions",
        "unverified_budget_exceeded",
    ] = "pending"
    verification_reason: str = ""

    @field_validator("estimated_start_sec", "estimated_end_sec", "timecode_sec")
    @classmethod
    def _check_finite_non_negative(cls, v: float) -> float:
        if not isinstance(v, (int, float)) or not math.isfinite(float(v)) or float(v) < 0.0:
            raise ValueError(f"Временная метка кандидата должна быть конечным неотрицательным числом, получено: {v}")
        return round(float(v), 2)


class EvidenceFrameModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_id: str = Field(min_length=2)
    pts: int
    timestamp_sec: float
    image_base64: str = Field(min_length=32)

    @field_validator("timestamp_sec")
    @classmethod
    def _check_finite_ts(cls, v: float) -> float:
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(float(v)) or float(v) < 0.0:
            raise ValueError(f"Недопустимое значение timestamp_sec доказательного кадра: {v}")
        return round(float(v), 2)

    @field_validator("image_base64")
    @classmethod
    def _check_valid_jpeg_b64(cls, v: str) -> str:
        try:
            raw = base64.b64decode(v, validate=True)
            with Image.open(io.BytesIO(raw)) as img:
                img.verify()
        except Exception as e:
            raise ValueError(f"Повреждённый image_base64 (не является валидным JPEG): {e}") from e
        return v


class RegulationBasisModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    short_title: str
    document: str
    clause: str
    url: str
    verified_date: str
    rule_summary: str
    applicability_conditions: List[str] = Field(min_length=1)
    mandatory_visual_indicators: List[str] = Field(min_length=1)
    inference_limitations: List[str] = Field(default_factory=list)
    enabled: bool
    disabled_reason: Optional[str] = None


class ConditionCheckModel(BaseModel):
    """Связь «условие — участник — объект — подтверждающий факт — кадры» одного события (ревью 0909, п. 4.2)."""

    model_config = ConfigDict(extra="forbid")

    condition_id: str
    condition: str
    status: Literal["true", "unverified_cv_hypothesis"]
    actor: Optional[str] = None
    target_object: str
    supporting_fact: str
    evidence_frame_ids: List[str] = Field(min_length=1)
    source: str
    structured_input: Optional[Dict[str, Any]] = None


_CHECK_COMPARE_KEYS = ("condition_id", "condition", "status", "actor", "target_object", "supporting_fact", "evidence_frame_ids")


class VerifiedViolationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    type: str = Field(min_length=5)
    timecode_sec: float
    description: str = Field(min_length=20)
    observed_facts: str = Field(min_length=10)
    evidence_frames: List[EvidenceFrameModel] = Field(min_length=1, max_length=3)
    evidence_source: Literal["vlm_verified", "cv_detector_unverified"] = EVIDENCE_SOURCE_VLM
    mandatory_conditions_confirmed: List[str] = Field(default_factory=list)
    condition_checks: List[ConditionCheckModel] = Field(default_factory=list)
    regulation: RegulationBasisModel
    candidate_id: Optional[str] = None
    episode_id: Optional[str] = None
    actor_description: Optional[str] = None
    cash_visible_on_desk: Optional[bool] = None
    cashier_left_room: Optional[bool] = None
    seal_check_performed: Optional[bool] = None
    recount_performed: Optional[bool] = None
    phone_in_hands: Optional[bool] = None
    safe_door_open: Optional[bool] = None
    safe_contents_visible: Optional[bool] = None
    client_present: Optional[bool] = None

    @field_validator("timecode_sec")
    @classmethod
    def _check_finite_timecode(cls, v: float) -> float:
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(float(v)) or float(v) < 0.0:
            raise ValueError(f"timecode_sec должен быть конечным неотрицательным числом, получено: {v}")
        return round(float(v), 2)

    @model_validator(mode="after")
    def _validate_rule_and_evidence(self) -> "VerifiedViolationModel":
        if self.code not in VIOLATION_CATALOG:
            raise ValueError(f"Неизвестный код нарушения '{self.code}' (отсутствует в VIOLATION_CATALOG).")
        cat_entry = VIOLATION_CATALOG[self.code]
        if not cat_entry.get("enabled", False):
            raise ValueError(
                f"Код правила '{self.code}' отключён (enabled=False) и не может входить в подтверждённые violations: "
                f"{cat_entry.get('disabled_reason')}"
            )
        if self.regulation.code != self.code or not self.regulation.enabled:
            raise ValueError(f"Несогласованность нормативного блока regulation для кода '{self.code}'.")
        if not self.regulation.url.startswith("https://adilet.zan.kz/rus/docs/"):
            raise ValueError(f"Некорректная нормативная ссылка: {self.regulation.url}")

        frame_ids = [ef.frame_id for ef in self.evidence_frames]
        entry_flags = {k: getattr(self, k) for k in PHYSICAL_FLAG_KEYS}

        # F01 / R02 / R05 / ревью 0909: условия правила вычисляются заново из связок «участник — действие — объект»
        # и сравниваются с опубликованными флагами, связями условий и списком подтверждённых условий.
        recomputed = evaluate_rule_conditions(
            code=self.code,
            observed_facts=self.observed_facts,
            structured_conditions=dict(entry_flags),
            evidence_frame_ids=frame_ids,
            evidence_source=self.evidence_source,
        )
        validate_observation_consistency(
            code=self.code,
            observed_facts=self.observed_facts,
            confirmed_conditions=self.mandatory_conditions_confirmed,
            structured_conditions=dict(entry_flags),
            evidence_frame_ids=frame_ids,
            evidence_source=self.evidence_source,
        )
        if self.evidence_source == EVIDENCE_SOURCE_VLM:
            if not self.mandatory_conditions_confirmed:
                raise ValueError(f"Для подтверждённого нарушения '{self.code}' отсутствует список подтверждённых условий.")
            if list(self.mandatory_conditions_confirmed) != list(recomputed["confirmed_conditions"]):
                raise ValueError(
                    f"Опубликованные mandatory_conditions_confirmed для '{self.code}' не совпадают с условиями, "
                    f"установленными по observed_facts."
                )
        elif self.mandatory_conditions_confirmed:
            raise ValueError(
                f"Гипотеза CV-детектора '{self.code}' не может содержать подтверждённых физических условий "
                f"(evidence_source=cv_detector_unverified)."
            )
        if entry_flags != recomputed["physical_flags"]:
            raise ValueError(
                f"Опубликованные физические флаги нарушения '{self.code}' {entry_flags} не совпадают с проверенными "
                f"условиями {recomputed['physical_flags']}."
            )
        published_checks = [{k: getattr(ch, k) for k in _CHECK_COMPARE_KEYS} for ch in self.condition_checks]
        expected_checks = [{k: ch[k] for k in _CHECK_COMPARE_KEYS} for ch in recomputed["condition_checks"]]
        if published_checks != expected_checks:
            raise ValueError(
                f"Опубликованные condition_checks нарушения '{self.code}' не совпадают с проверенными связями "
                f"«условие — участник — объект — факт»."
            )

        # F04: Проверка временной связности timecode_sec и доказательных кадров
        min_diff = min(abs(ef.timestamp_sec - self.timecode_sec) for ef in self.evidence_frames)
        if min_diff > 4.0:
            raise ValueError(
                f"Время нарушения '{self.code}' ({self.timecode_sec}s) не связано с доказательными кадрами "
                f"({[ef.timestamp_sec for ef in self.evidence_frames]}, min_diff={min_diff:.2f}s > 4.0s)."
            )
        return self


class DiagnosticObservationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    timecode_sec: float
    observed_facts: str
    referenced_frame_ids: List[str] = Field(default_factory=list)
    reason_not_violation: str


class VLMRetryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: str
    attempt: int
    initial_error: str
    retry_outcome: Literal[
        "recovered",
        "failed",
        "recovered_semantic_events_rejected",
        "recovered_domain_rechecked",
    ]


class WindowDomainDecisionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: str
    is_banking_cash_room_scene: bool
    domain_status: Literal[
        "banking_or_workstation_scene",
        "office_workstation_interior",
        "rejected_non_banking_domain",
    ]
    scene_description: str
    domain_rejection_reason: Optional[str] = None
    shown_frame_ids: List[str] = Field(default_factory=list)


class AnalysisDiagnosticsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pipeline_mode: Literal["hybrid", "vlm_only", "cv_only"]
    candidates: List[CandidateEvent] = Field(default_factory=list)
    diagnostic_observations: List[DiagnosticObservationModel] = Field(default_factory=list)
    visibility_limitations: List[str] = Field(default_factory=list)
    vlm_retry_log: List[VLMRetryRecord] = Field(default_factory=list)
    window_domain_decisions: List[WindowDomainDecisionModel] = Field(default_factory=list)
    analysis_incomplete: bool = False
    incomplete_reason: Optional[str] = None

    @model_validator(mode="after")
    def _validate_no_pending_candidates(self) -> "AnalysisDiagnosticsModel":
        if not self.analysis_incomplete:
            pending = [c.candidate_id for c in self.candidates if c.verification_status == "pending"]
            if pending:
                raise ValueError(
                    f"Обнаружены непроверенные кандидаты со статусом 'pending' при завершённом анализе: {pending}"
                )
        return self


class AnalysisResultModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["violation", "no_violation"]
    violations: List[VerifiedViolationModel]
    meta: Dict[str, Any]

    @model_validator(mode="after")
    def _validate_cross_field_invariants(self, info: ValidationInfo) -> "AnalysisResultModel":
        # 1. Инвариант verdict <-> violations (Раздел 8.2 п. 94)
        if self.verdict == "violation" and len(self.violations) == 0:
            raise ValueError("Инвариант нарушен: при verdict='violation' список violations не может быть пустым.")
        if self.verdict == "no_violation" and len(self.violations) > 0:
            raise ValueError("Инвариант нарушен: при verdict='no_violation' список violations должен быть пустым.")

        # 2. Инвариант границ временной шкалы без молчаливого обрезания (Раздел 6.1 п. 71–72, Раздел 8.2 п. 97)
        duration_sec = float(self.meta.get("duration_sec", 0.0))
        if not math.isfinite(duration_sec) or duration_sec <= 0.0:
            raise ValueError(f"Некорректная длительность видео в meta.duration_sec: {duration_sec}")

        allowed_frame_ids: Optional[Set[str]] = None
        if info.context and isinstance(info.context, dict):
            allowed_frame_ids = info.context.get("allowed_frame_ids")

        for v in self.violations:
            if v.evidence_source == EVIDENCE_SOURCE_CV and self.meta.get("pipeline_mode") != "cv_only":
                raise ValueError(
                    f"Нарушение {v.code} основано только на непроверенной гипотезе CV-детектора "
                    f"(evidence_source={EVIDENCE_SOURCE_CV}); такие записи допустимы лишь в диагностическом режиме cv_only."
                )
            if v.timecode_sec < 0.0 or v.timecode_sec > duration_sec:
                raise ValueError(
                    f"Таймкод нарушения {v.code} ({v.timecode_sec}s) выходит за пределы длительности видео [0, {duration_sec}s]."
                )
            for ef in v.evidence_frames:
                if ef.timestamp_sec < 0.0 or ef.timestamp_sec > duration_sec:
                    raise ValueError(
                        f"Таймкод доказательного кадра {ef.frame_id} ({ef.timestamp_sec}s) выходит за пределы [0, {duration_sec}s]."
                    )
                if allowed_frame_ids is not None and ef.frame_id not in allowed_frame_ids:
                    raise ValueError(
                        f"Кадр доказательства '{ef.frame_id}' не принадлежит множеству реально декодированных кадров видео."
                    )

        diag = self.meta.get("diagnostics")
        if isinstance(diag, dict) and not diag.get("analysis_incomplete", False):
            for c in diag.get("candidates") or []:
                if isinstance(c, dict) and c.get("verification_status") == "pending":
                    raise ValueError(
                        f"Непроверенный кандидат {c.get('candidate_id')} остался в статусе 'pending'."
                    )

        return self


# --- СХЕМЫ ДЛЯ STRUCTURED GENERATION В MLX-VLM 0.7.8 (llguidance JSON Schema) ---


def build_vlm_output_json_schema(allowed_frame_ids: List[str]) -> Dict[str, Any]:
    """
    Формирует строгую JSON Schema для `mlx_vlm.structured.build_json_schema_logits_processor`.
    Ограничивает:
    - `is_banking_cash_room_scene`: булев флаг банковского кассового помещения (для отсечения небанковских сцен MEVA),
    - `code` в `confirmed_events`: только разрешёнными публичными правилами (`enabled=True`),
    - `code` в `diagnostic_observations`: отключёнными правилами (`enabled=False`) из `VIOLATION_CATALOG`,
    - `evidence_frame_ids`: только реально показанными модели `frame_id` на текущей раскадровке,
    - `candidate_id` и структурные поля условий (`phone_in_hands`, `cash_visible_on_desk`, `cashier_left_room`,
      `seal_check_performed`, `recount_performed`, `safe_door_open`, `safe_contents_visible`, `client_present`, `condition_checks`),
    - `maxLength` и `maxItems`: гарантирует компактный валидный JSON в пределах бюджета `max_tokens`.
    """
    enabled_codes = sorted(get_enabled_catalog().keys())
    disabled_codes = sorted(k for k, v in VIOLATION_CATALOG.items() if not v.get("enabled", False))
    if not disabled_codes:
        disabled_codes = sorted(VIOLATION_CATALOG.keys())
    frame_id_enum = sorted(set(allowed_frame_ids)) if allowed_frame_ids else ["F0000"]

    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "scene_description",
            "is_banking_cash_room_scene",
            "confirmed_events",
            "rejected_hypothesis_codes",
            "diagnostic_observations",
            "visibility_limitations",
        ],
        "properties": {
            "scene_description": {
                "type": "string",
                "minLength": 10,
                "maxLength": 110,
            },
            "is_banking_cash_room_scene": {
                "type": "boolean",
            },
            "confirmed_events": {
                "type": "array",
                "maxItems": 2,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "code",
                        "timecode_sec",
                        "observed_facts",
                        "evidence_frame_ids",
                    ],
                    "properties": {
                        "code": {
                            "type": "string",
                            "enum": enabled_codes,
                        },
                        "timecode_sec": {
                            "type": "number",
                            "minimum": 0.0,
                        },
                        "observed_facts": {
                            "type": "string",
                            "minLength": 12,
                            "maxLength": 115,
                        },
                        "evidence_frame_ids": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 3,
                            "items": {
                                "type": "string",
                                "enum": frame_id_enum,
                            },
                        },
                    },
                },
            },
            "rejected_hypothesis_codes": {
                "type": "array",
                "maxItems": 2,
                "items": {
                    "type": "string",
                    "maxLength": 45,
                },
            },
            "diagnostic_observations": {
                "type": "array",
                "maxItems": 2,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "code",
                        "timecode_sec",
                        "observed_facts",
                        "referenced_frame_ids",
                    ],
                    "properties": {
                        "code": {
                            "type": "string",
                            "enum": disabled_codes,
                        },
                        "timecode_sec": {
                            "type": "number",
                            "minimum": 0.0,
                        },
                        "observed_facts": {
                            "type": "string",
                            "minLength": 10,
                            "maxLength": 100,
                        },
                        "referenced_frame_ids": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 3,
                            "items": {
                                "type": "string",
                                "enum": frame_id_enum,
                            },
                        },
                    },
                },
            },
            "visibility_limitations": {
                "type": "array",
                "maxItems": 2,
                "items": {
                    "type": "string",
                    "maxLength": 90,
                },
            },
        },
    }

