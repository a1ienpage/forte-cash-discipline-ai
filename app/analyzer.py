"""
Главный оркестратор видеоаналитики кассовой дисциплины ForteBank (Datathon 26, Версия 2.0).

Реализует целевую последовательность Раздела 5 ТЗ:
1. Проверка входного файла и декодирование через PyAV (`PyAVVideoDecoder.decode_timeline`) с построением
   точной временной шкалы по `PTS * time_base` и нормализацией начального смещения потока.
2. Масштабно-инвариантный обзор наблюдений и генерация непроверенных гипотез (`CVSceneAnalyzer.analyze_timeline`)
   в нормализованных координатах `[0, 1]` без сценарных/ракурсных/длительностных эвристик.
3. Первичный обзорный проход локальной VLM (`Qwen2.5-VL-7B-Instruct-4bit`, Apache-2.0) по всей длине видео `[0, duration_sec]`
   со строгим ограничением генерации по JSON Schema (`mlx_vlm/structured.py`).
4. Плотный уточняющий проход VLM (`sample_dense_window`) по локальным временным участкам кандидатов для:
   - отклонения ложных гипотез CV (`rejected_by_vlm`),
   - уточнения `timecode_sec` по реальным PTS-меткам плотного окна,
   - выбора 1–3 доказательных кадров разных значимых фаз действия (`evidence_frame_ids`).
5. Проверка условий применимости правил по нормативному справочнику (`app/regulations.py`) с отделением
   подтверждённых публичных нарушений (`enabled=True`) от диагностических наблюдений (`enabled=False`).
6. Полная межполевая и семантическая валидация результата через Pydantic v2 (`AnalysisResultModel`).
"""

import os
import time
from typing import Any, Dict, List, Literal, Optional

import numpy as np

from app.cv_detector import CVSceneAnalyzer
from app.regulations import (
    EVIDENCE_SOURCE_CV,
    EVIDENCE_SOURCE_VLM,
    VIOLATION_CATALOG,
    _split_clauses_with_actors,
    format_violation_entry,
    get_enabled_catalog,
)
from app.schemas import (
    AnalysisDiagnosticsModel,
    AnalysisResultModel,
    CandidateEvent,
    DiagnosticObservationModel,
    VLMRetryRecord,
    WindowDomainDecisionModel,
)
from app.video_decoder import DecodedFrame, PyAVVideoDecoder, VideoTimelineBundle
from app.vlm_engine import (
    APACHE_2_MODEL_ID,
    APACHE_2_MODEL_REVISION,
    MODE_POLICIES,
    LocalVLMEngine,
    SemanticRejectionError,
)

SEQUENTIAL_PHASE_CODES = {
    "unattended_cash_on_desk_on_exit",
    "unattended_collection_bag_on_desk",
    "safe_left_open_on_exit",
    "unattended_documents_stamps_on_exit",
    "counting_in_client_absence",
    "unattended_previous_client_cash",
    "multiple_clients_in_cash_room",
    "collection_bag_accepted_without_check",
}


class CashDisciplineAnalyzer:
    """
    Двухпроходный локальный анализатор видеозаписей кассовой дисциплины без предзаписанных ответов и сценарных эвристик.
    Поддерживает три явно переключаемых режима пайплайна для абляционных экспериментов (Раздел 18.2):
    - `hybrid` (по умолчанию): CV-наблюдения/гипотезы + двухпроходная верификация и поиск VLM;
    - `vlm_only`: двухпроходный анализ VLM (обзор всего видео + плотное уточнение) без передачи CV-гипотез;
    - `cv_only`: диагностический прогон только CV-детектора без вызова VLM.
    """

    def __init__(self, default_vlm_mode: str = "auto"):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.cv_analyzer = CVSceneAnalyzer(yolo_weights="yolo11s.pt")
        self.vlm_engine = LocalVLMEngine(default_mode=default_vlm_mode)

    @staticmethod
    def _select_coarse_storyboard_frames(
        bundle: VideoTimelineBundle,
        num_frames: int,
        max_interval_sec: float = 4.0,
    ) -> List[DecodedFrame]:
        """
        Выбирает до `num_frames` декодированных кадров с гарантированным верхним пределом шага
        `max_interval_sec` между соседними кадрами окна (Раздел 6.2, F02).
        Для длинных роликов полное покрытие всей шкалы `[0, duration_sec]` обеспечивается
        последовательностью перекрывающихся окон `_build_coarse_overview_windows`.
        """
        windows = CashDisciplineAnalyzer._build_coarse_overview_windows(
            bundle=bundle,
            frames_per_window=num_frames,
            max_interval_sec=max_interval_sec,
        )
        return windows[0] if windows else list(bundle.coarse_frames[:num_frames])

    @staticmethod
    def _build_coarse_overview_windows(
        bundle: VideoTimelineBundle,
        frames_per_window: int,
        max_interval_sec: float = 4.0,
    ) -> List[List[DecodedFrame]]:
        """
        Строит последовательность перекрывающихся обзорных окон для Pass 1 VLM так, чтобы:
        1. Покрыть всю временную шкалу `[0, duration_sec]` от первого до последнего кадра;
        2. Межкадровый интервал внутри каждого окна VLM строго не превышал `max_interval_sec` (F02, R07).
        """
        coarse = bundle.coarse_frames
        if not coarse:
            return []
        if len(coarse) == 1:
            return [list(coarse)]

        # Строим опорную сетку кадров VLM, выбирая последний кадр с шагом <= max_interval_sec
        vlm_grid: List[DecodedFrame] = [coarse[0]]
        cur_idx = 0
        n_coarse = len(coarse)
        while cur_idx < n_coarse - 1:
            cur_t = float(coarse[cur_idx].timestamp_sec)
            best_next = cur_idx + 1
            probe = cur_idx + 1
            while probe < n_coarse and (float(coarse[probe].timestamp_sec) - cur_t) <= (max_interval_sec + 1e-6):
                best_next = probe
                probe += 1
            vlm_grid.append(coarse[best_next])
            cur_idx = best_next

        if len(vlm_grid) <= frames_per_window:
            # Если вся сетка помещается в одно окно, пробуем равномерно дополнить до frames_per_window при сохранении шага <= max_interval_sec
            if len(coarse) <= frames_per_window:
                max_gap_all = max(
                    (float(b.timestamp_sec) - float(a.timestamp_sec) for a, b in zip(coarse, coarse[1:])),
                    default=0.0,
                )
                if max_gap_all <= max_interval_sec + 1e-6:
                    return [list(coarse)]
            idxs = np.linspace(0, len(coarse) - 1, num=min(frames_per_window, len(coarse)), dtype=int)
            win: List[DecodedFrame] = []
            seen = set()
            for i in idxs:
                df = coarse[int(i)]
                if df.frame_id not in seen:
                    seen.add(df.frame_id)
                    win.append(df)
            max_gap_win = max(
                (float(b.timestamp_sec) - float(a.timestamp_sec) for a, b in zip(win, win[1:])),
                default=0.0,
            )
            if max_gap_win <= max_interval_sec + 1e-6:
                return [win]
            return [vlm_grid]

        windows: List[List[DecodedFrame]] = []
        step = max(1, frames_per_window - 1)
        for start_idx in range(0, len(vlm_grid), step):
            chunk = vlm_grid[start_idx : start_idx + frames_per_window]
            if len(chunk) < 2 and windows:
                break
            windows.append(chunk)
            if start_idx + frames_per_window >= len(vlm_grid):
                break
        return windows

    @staticmethod
    def _ensure_multiphase_evidence_ids(
        code: str,
        primary_frame_ids: List[str],
        window_frames: List[DecodedFrame],
        target_timecode_sec: float,
    ) -> List[str]:
        """
        Гарантирует сохранение ВСЕХ выбранных моделью `primary_frame_ids` (F04) и при наличии свободных
        слотов (до 3 кадров) дополняет их контекстными кадрами начала/конца фазы из `window_frames`
        в хронологическом порядке (Раздел 9 п. 106).
        """
        if not window_frames:
            return list(dict.fromkeys(primary_frame_ids))[:3]

        valid_window_ids = {f.frame_id for f in window_frames}
        chosen = list(dict.fromkeys(fid for fid in primary_frame_ids if fid in valid_window_ids))

        if not chosen:
            closest = min(window_frames, key=lambda f: abs(f.timestamp_sec - target_timecode_sec))
            chosen = [closest.frame_id]

        if code in SEQUENTIAL_PHASE_CODES and len(chosen) < 3 and len(window_frames) >= 3:
            keep_ids = list(chosen[:3])
            pre_fid = window_frames[0].frame_id
            post_fid = window_frames[-1].frame_id
            mid_fid = min(window_frames, key=lambda f: abs(f.timestamp_sec - target_timecode_sec)).frame_id

            for candidate_fid in (pre_fid, post_fid, mid_fid):
                if len(keep_ids) >= 3:
                    break
                if candidate_fid not in keep_ids:
                    keep_ids.append(candidate_fid)

            keep_set = set(keep_ids)
            ordered = [f.frame_id for f in window_frames if f.frame_id in keep_set]
            # Гарантируем, что ни один изначально выбранный кадр chosen не потерян
            for c_fid in chosen[:3]:
                if c_fid not in ordered:
                    ordered.append(c_fid)
            return ordered[:3]

        return chosen[:3]

    @staticmethod
    def _log_semantic_rejections(
        vlm_out: Dict[str, Any],
        candidates_log: List[CandidateEvent],
        origin: str,
    ) -> None:
        """
        Фиксирует в диагностике события VLM, отброшенные семантической проверкой (условие правила не установлено
        связкой «участник — действие — объект»), со статусом `rejected_missing_conditions` (ревью 0909, п. 4.1).
        """
        for idx_r, rej in enumerate(vlm_out.get("semantically_rejected_events") or []):
            try:
                tc = float(rej["timecode_sec"])
                candidates_log.append(
                    CandidateEvent(
                        candidate_id=f"VLM_SEMANTIC_REJECT_{len(candidates_log) + 1:02d}",
                        code=rej["code"],
                        observed_facts=str(rej.get("observed_facts") or "—")[:400] or "—",
                        referenced_frame_ids=list(rej.get("evidence_frame_ids") or ["F0000"]),
                        estimated_start_sec=max(0.0, tc - 1.0),
                        estimated_end_sec=tc + 1.0,
                        timecode_sec=tc,
                        origin=origin,
                        verification_status="rejected_missing_conditions",
                        verification_reason=str(rej.get("reason") or "")[:600],
                    )
                )
            except Exception:
                continue

    @staticmethod
    def _extract_actor_signature(obs_text: str, actor_desc: Optional[str] = None) -> set:
        """
        Извлекает нормализованные различительные признаки САМОГО участника-сотрудника (порядковый номер,
        пространственное положение, цвет одежды, относительное «другой/ещё один») только из клауз, где этот
        сотрудник является подлежащим, и из `actor_description` (R06; ревью 0909, п. 4.3).
        Признаки других людей («слева от него стоит клиент») в подпись сотрудника не попадают.
        """
        import re

        actor_parts: List[str] = []
        clauses = _split_clauses_with_actors(obs_text or "")
        cashier_clauses = [c["text"] for c in clauses if c["actor"] == "cashier"]
        if cashier_clauses:
            actor_parts.extend(cashier_clauses)
        elif not any(c["actor"] in ("client", "other_person") for c in clauses):
            actor_parts.append((obs_text or "").lower())
        if actor_desc:
            actor_parts.append(str(actor_desc).lower())
        combined = " ".join(actor_parts)
        # Положение относительно другого человека описывает не самого сотрудника
        combined = re.sub(r"\b(?:слева|справа)\s+от\s+(?:него|нее|неё|них|\w+)", " ", combined)

        actor_noun = r"(?:кассир\w*|сотрудни\w*|работни\w*|операционист\w*|специалист\w*)"
        sig = set()
        for m in re.finditer(r"\b(перв\w*|втор\w*|трет\w*)\s+" + actor_noun, combined):
            sig.add("ord:" + {"перв": "1", "втор": "2", "трет": "3"}[m.group(1)[:4]])
        for m in re.finditer(actor_noun + r"\s*(?:№|номер)\s*(\d+)", combined):
            sig.add("ord:" + m.group(1))
        if re.search(r"\b(?:друг\w+|ещ[её]\s+од\w+|соседн\w+)\s+" + actor_noun + r"|\bотдельно\b", combined):
            sig.add("rel:other")
        if re.search(r"\bслева\b|\bлев(?:ый|ая|ом|ой)\b|в\s+левой\s+части", combined):
            sig.add("pos:left")
        if re.search(r"\bсправа\b|\bправ(?:ый|ая|ом|ой)\b|в\s+правой\s+части", combined):
            sig.add("pos:right")
        if re.search(r"на\s+переднем\s+плане|\bближн\w+", combined):
            sig.add("plane:front")
        if re.search(r"на\s+заднем\s+плане|\bдальн\w+", combined):
            sig.add("plane:back")
        for color, stems in (
            ("red", ("красн", "бордов")),
            ("blue", ("син", "голуб")),
            ("white", ("бел", "светл")),
            ("black", ("черн", "чёрн", "темн", "тёмн")),
            ("green", ("зелен", "зелён")),
            ("yellow", ("желт", "жёлт")),
        ):
            if any(re.search(r"\b" + s, combined) for s in stems):
                sig.add("color:" + color)
        return sig

    @staticmethod
    def _actor_signatures_conflict(sig_a: set, sig_b: set, use_relative: bool = False) -> bool:
        """
        Возвращает True, если признаки двух описаний явно указывают на разных людей: разные порядковые номера,
        противоположное положение, непересекающиеся цвета одежды. При `use_relative=True` относительный
        маркер «другой/ещё один/отдельно», присутствующий только в одном описании, также означает другого человека.
        """
        for prefix in ("ord:", "pos:", "plane:"):
            a = {x for x in sig_a if x.startswith(prefix)}
            b = {x for x in sig_b if x.startswith(prefix)}
            if a and b and a.isdisjoint(b):
                return True
        colors_a = {x for x in sig_a if x.startswith("color:")}
        colors_b = {x for x in sig_b if x.startswith("color:")}
        if colors_a and colors_b and colors_a.isdisjoint(colors_b):
            return True
        if use_relative and (("rel:other" in sig_a) != ("rel:other" in sig_b)):
            return True
        return False

    @classmethod
    def _merge_same_episode_events(cls, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Дедупликация повторных наблюдений одного и того же физического эпизода (Раздел 9 п. 104–105, R06; ревью 0909, п. 4.3):
        - события РАЗНЫХ типов (`code1 != code2`) НИКОГДА не объединяются;
        - идентичность участника сравнивается между двумя событиями (порядковый номер, положение, одежда,
          относительный маркер «другой»); одно и то же порядковое слово в обоих описаниях («второй кассир»)
          объединению не мешает, а разные признаки участника всегда сохраняют два события;
        - одинаковый явный `episode_id` при совместимых участниках и `|dt| <= 10 с` объединяет наблюдения даже при
          разных кадрах; разные явные `episode_id` из одного окна означают разные эпизоды;
        - без `episode_id` объединяются наблюдения из соседних окон с общим первичным кадром или общим `candidate_id`
          (`|dt| <= 4 с`); разные кандидаты одного плотного окна остаются отдельными событиями.
        Идентификаторы гипотезы (`candidate_id`), участника (`actor_description`) и физического эпизода (`episode_id`)
        сохраняются раздельно.
        """
        sorted_evs = sorted(events, key=lambda x: (float(x["timecode_sec"]), x["code"]))
        merged: List[Dict[str, Any]] = []

        for ev in sorted_evs:
            matched = None
            ev_primary_fids = set(ev.get("primary_evidence_frame_ids") or ev.get("evidence_frame_ids") or [])
            ev_cand_id = ev.get("candidate_id")
            ev_win_idx = ev.get("window_idx")
            ev_episode = (ev.get("episode_id") or "").strip() or None
            ev_sig = cls._extract_actor_signature(ev.get("observed_facts") or "", ev.get("actor_description"))

            for ex in merged:
                if ex["code"] != ev["code"]:
                    continue
                dt = abs(float(ex["timecode_sec"]) - float(ev["timecode_sec"]))
                if dt > 10.0:
                    continue
                ex_sig = cls._extract_actor_signature(ex.get("observed_facts") or "", ex.get("actor_description"))
                if cls._actor_signatures_conflict(ev_sig, ex_sig, use_relative=True):
                    continue

                ex_cand_id = ex.get("candidate_id")
                ex_win_idx = ex.get("window_idx")
                ex_episode = (ex.get("episode_id") or "").strip() or None
                ex_primary_fids = set(ex.get("primary_evidence_frame_ids") or ex.get("evidence_frame_ids") or [])
                same_cand = bool(ev_cand_id and ex_cand_id and ev_cand_id == ex_cand_id)
                shared_primary = bool(ev_primary_fids & ex_primary_fids)
                same_window = ev_win_idx is not None and ex_win_idx is not None and ev_win_idx == ex_win_idx

                if ev_episode and ex_episode:
                    if ev_episode == ex_episode:
                        matched = ex
                        break
                    if same_window:
                        continue

                if dt > 4.0:
                    continue
                if same_window:
                    # Два события одного вызова плотного окна — одно и то же только при общем кандидате и кадре
                    if not (same_cand and shared_primary):
                        continue
                elif not shared_primary and not same_cand:
                    continue

                matched = ex
                break

            if matched is None:
                merged.append(dict(ev))
            else:
                comb_fids = list(dict.fromkeys(matched["evidence_frame_ids"] + ev["evidence_frame_ids"]))[:3]
                matched["evidence_frame_ids"] = comb_fids
                comb_prim = list(
                    dict.fromkeys(
                        (matched.get("primary_evidence_frame_ids") or matched["evidence_frame_ids"])
                        + (ev.get("primary_evidence_frame_ids") or ev["evidence_frame_ids"])
                    )
                )
                matched["primary_evidence_frame_ids"] = comb_prim
                if not matched.get("episode_id") and ev.get("episode_id"):
                    matched["episode_id"] = ev.get("episode_id")
                if not matched.get("actor_description") and ev.get("actor_description"):
                    matched["actor_description"] = ev.get("actor_description")
                merged_ids = list(matched.get("merged_candidate_ids") or [matched.get("candidate_id")])
                if ev.get("candidate_id") and ev.get("candidate_id") not in merged_ids:
                    merged_ids.append(ev.get("candidate_id"))
                matched["merged_candidate_ids"] = [x for x in merged_ids if x]

        return merged

    def _run_vlm_pass_guarded(self, **kwargs: Any):
        """
        Вызывает проход VLM. Если движок вернул типизированный отказ семантической проверки события
        (`SemanticRejectionError`) вместо собственной обработки повтора, окно считается не подтвердившим ни одного
        события (событие не публикуется), а факт отказа фиксируется в `vlm_retry_log` (ревью 0909, п. 4.1).
        Структурные ошибки ответа по-прежнему переводят задачу в `failed`.
        """
        try:
            return self.vlm_engine.run_vlm_pass(**kwargs)
        except SemanticRejectionError as sem_err:
            empty = {
                "is_banking_cash_room_scene": True,
                "raw_model_is_banking_scene": True,
                "domain_status": "banking_or_workstation_scene",
                "domain_rejection_reason": None,
                "scene_description": "Семантический отказ при проверке условий события",
                "visibility_limitations": [],
                "rejected_hypothesis_codes": [],
                "confirmed_events": [],
                "diagnostic_observations": [],
                "semantically_rejected_events": [],
            }
            record = VLMRetryRecord(
                stage=str(kwargs.get("stage_name") or "vlm_pass"),
                attempt=1,
                initial_error=str(sem_err)[:1000],
                retry_outcome="recovered_semantic_events_rejected",
            )
            return empty, [record], 1

    def analyze_video(
        self,
        video_path: str,
        original_filename: str = "",
        vlm_mode: str = "auto",
        invoke_vlm: bool = True,
        pipeline_mode: Optional[Literal["hybrid", "vlm_only", "cv_only"]] = None,
    ) -> Dict[str, Any]:
        """
        Выполняет полный цикл анализа видеозаписи по Разделам 5–10 ТЗ.
        Любая неисправимая техническая ошибка декодирования, отсутствия весов или нарушения схемы
        пробрасывается наверх и переводит задачу в статус `failed` (никогда не превращается в `no_violation`).
        """
        t_start = time.time()

        if pipeline_mode is None:
            pipeline_mode = "hybrid" if invoke_vlm else "cv_only"
        elif pipeline_mode == "cv_only":
            invoke_vlm = False

        # Шаг 1. Декодирование через PyAV и построение временной шкалы по PTS * time_base
        t_decode_start = time.time()
        target_interval = 0.75 if vlm_mode == "deep" else 1.0
        bundle = PyAVVideoDecoder.decode_timeline(
            video_path=video_path,
            target_interval_sec=target_interval,
        )
        decode_elapsed_sec = round(time.time() - t_decode_start, 3)

        # Шаг 2. Пространственно-временной анализ наблюдений и кандидатов CV (в нормализованных координатах [0, 1])
        cv_info = self.cv_analyzer.analyze_timeline(
            bundle=bundle,
            run_yolo=(pipeline_mode in ("hybrid", "cv_only")),
        )
        cv_candidates: List[CandidateEvent] = (
            cv_info["candidates"] if pipeline_mode in ("hybrid", "cv_only") else []
        )
        visibility_limitations: List[str] = list(cv_info["visibility_limitations"])

        resolved_mode = self.vlm_engine.resolve_mode(
            requested_mode=vlm_mode,
            duration_sec=bundle.duration_sec,
            cv_candidates_count=len(cv_candidates),
        )
        policy = MODE_POLICIES[resolved_mode]

        all_candidates_log: List[CandidateEvent] = [c.model_copy(deep=True) for c in cv_candidates]
        diagnostic_observations: List[DiagnosticObservationModel] = []
        vlm_retry_log: List[VLMRetryRecord] = []
        window_domain_decisions: List[WindowDomainDecisionModel] = []
        vlm_calls = 0
        vlm_inference_sec = 0.0
        allowed_all_fids = set((bundle.frame_pts_map or {}).keys()) | set(bundle.all_decoded_frames.keys())

        # Если видео технически валидное, но полностью однотонное/пустое (AT-01) — не придумываем события
        if cv_info["is_monotone_or_empty"]:
            elapsed_sec = round(time.time() - t_start, 2)
            diagnostics = AnalysisDiagnosticsModel(
                pipeline_mode=pipeline_mode,
                candidates=[],
                diagnostic_observations=[],
                visibility_limitations=visibility_limitations,
                vlm_retry_log=[],
                window_domain_decisions=[],
                analysis_incomplete=False,
            )
            empty_result = {
                "verdict": "no_violation",
                "violations": [],
                "meta": {
                    "filename": original_filename or os.path.basename(video_path),
                    "duration_sec": round(bundle.duration_sec, 2),
                    "resolution": f"{bundle.width}x{bundle.height}",
                    "fps": bundle.nominal_fps,
                    "codec_name": bundle.codec_name,
                    "time_base": bundle.time_base_str,
                    "start_offset_sec": bundle.start_offset_sec,
                    "offset_normalization_rule": bundle.offset_normalization_rule,
                    "is_vfr_detected": bundle.is_vfr_detected,
                    "cv_max_observation_interval_sec": bundle.max_observation_interval_sec,
                    "vlm_max_observation_interval_sec": None,
                    "max_observation_interval_sec": bundle.max_observation_interval_sec,
                    "vlm_coverage_complete": False,
                    "short_action_limitation_note": bundle.short_action_limitation_note,
                    "decode_time_sec": decode_elapsed_sec,
                    "vlm_inference_time_sec": 0.0,
                    "processing_time_sec": elapsed_sec,
                    "requested_mode": vlm_mode,
                    "resolved_mode": resolved_mode,
                    "pipeline_mode": pipeline_mode,
                    "vlm_model": APACHE_2_MODEL_ID if invoke_vlm else "cv_only",
                    "vlm_revision": APACHE_2_MODEL_REVISION,
                    "vlm_license": "Apache-2.0",
                    "vlm_calls": 0,
                    "max_persons_detected": 0,
                    "yolo_objects_detected": [],
                    "lights_off_detected": False,
                    "diagnostics": diagnostics.model_dump(),
                },
            }
            AnalysisResultModel.model_validate(
                empty_result,
                context={"allowed_frame_ids": allowed_all_fids},
            )
            return empty_result

        confirmed_raw_events: List[Dict[str, Any]] = []
        vlm_max_interval_sec: Optional[float] = None
        vlm_coverage_complete: bool = False

        if invoke_vlm and pipeline_mode in ("hybrid", "vlm_only"):
            t_vlm_start = time.time()
            enabled_cat = get_enabled_catalog()

            # Отделяем диагностические кандидаты отключённых правил (enabled=False) от публичных гипотез (enabled=True)
            for c_cand in all_candidates_log:
                if c_cand.origin == "cv_hypothesis" and c_cand.code not in enabled_cat:
                    c_cand.verification_status = "diagnostic_only_disabled_rule"
                    reason = VIOLATION_CATALOG.get(c_cand.code, {}).get("disabled_reason") or "Отключённое правило СОП"
                    c_cand.verification_reason = reason
                    diagnostic_observations.append(
                        DiagnosticObservationModel(
                            code=c_cand.code,
                            timecode_sec=c_cand.timecode_sec,
                            observed_facts=c_cand.observed_facts,
                            referenced_frame_ids=c_cand.referenced_frame_ids,
                            reason_not_violation=reason,
                        )
                    )

            enabled_cv_candidates = [c for c in all_candidates_log if c.origin == "cv_hypothesis" and c.code in enabled_cat]

            # Шаг 3. Первичный обзорный проход VLM по перекрывающимся окнам всей длины видео [0, duration_sec] (Раздел 6.2, F02)
            overview_windows = self._build_coarse_overview_windows(
                bundle=bundle,
                frames_per_window=int(policy["coarse_num_frames"]),
                max_interval_sec=float(policy.get("coarse_max_interval_sec", 4.0)),
            )
            max_coarse_windows = int(policy.get("coarse_max_windows", 6))
            executed_overview_windows = overview_windows[:max_coarse_windows]
            vlm_coverage_complete = len(executed_overview_windows) == len(overview_windows)

            all_vlm_shown_times: List[float] = []
            coarse_confirmed_all: List[Dict[str, Any]] = []

            for ow_idx, coarse_frames in enumerate(executed_overview_windows):
                for cf in coarse_frames:
                    all_vlm_shown_times.append(float(cf.timestamp_sec))

                w_start = coarse_frames[0].timestamp_sec
                w_end = coarse_frames[-1].timestamp_sec
                win_cv_hyps = None
                if pipeline_mode == "hybrid":
                    win_cv_hyps = [
                        c
                        for c in enabled_cv_candidates
                        if (w_start - 2.0) <= c.timecode_sec <= (w_end + 2.0)
                    ]

                coarse_storyboard = PyAVVideoDecoder.build_annotated_storyboard(
                    frames=coarse_frames,
                    tile_size=policy["coarse_tile_size"],
                    cols=3,
                )
                stage_label = (
                    "pass1_full_video_coarse_overview"
                    if len(executed_overview_windows) == 1
                    else f"pass1_coarse_window_{ow_idx+1}_of_{len(executed_overview_windows)}"
                )
                coarse_out, retries_p1, calls_p1 = self._run_vlm_pass_guarded(
                    stage_name=stage_label,
                    storyboard_img=coarse_storyboard,
                    shown_frames=coarse_frames,
                    duration_sec=bundle.duration_sec,
                    cv_hypotheses=win_cv_hyps,
                    max_tokens=policy["max_tokens"],
                )
                vlm_calls += calls_p1
                vlm_retry_log.extend(retries_p1)
                window_domain_decisions.append(
                    WindowDomainDecisionModel(
                        stage=stage_label,
                        is_banking_cash_room_scene=bool(coarse_out.get("is_banking_cash_room_scene", True)),
                        domain_status=str(coarse_out.get("domain_status") or "banking_or_workstation_scene"),
                        scene_description=str(coarse_out.get("scene_description") or "—"),
                        domain_rejection_reason=coarse_out.get("domain_rejection_reason"),
                        shown_frame_ids=[f.frame_id for f in coarse_frames],
                    )
                )

                for lim in coarse_out.get("visibility_limitations", []):
                    if lim not in visibility_limitations:
                        visibility_limitations.append(lim)

                for dob in coarse_out.get("diagnostic_observations", []):
                    code = dob["code"]
                    reason = VIOLATION_CATALOG[code].get("disabled_reason") or "Диагностическое наблюдение"
                    diagnostic_observations.append(
                        DiagnosticObservationModel(
                            code=code,
                            timecode_sec=dob["timecode_sec"],
                            observed_facts=dob["observed_facts"],
                            referenced_frame_ids=dob["referenced_frame_ids"],
                            reason_not_violation=reason,
                        )
                    )

                self._log_semantic_rejections(coarse_out, all_candidates_log, origin="vlm_coarse")

                for c_ev in coarse_out.get("confirmed_events", []):
                    ev_key = (c_ev.get("code"), tuple(c_ev.get("evidence_frame_ids", [])))
                    if not any((x.get("code"), tuple(x.get("evidence_frame_ids", []))) == ev_key for x in coarse_confirmed_all):
                        coarse_confirmed_all.append(c_ev)

            sorted_vlm_times = sorted(set(all_vlm_shown_times))
            if len(sorted_vlm_times) >= 2:
                vlm_gaps = [
                    sorted_vlm_times[i] - sorted_vlm_times[i - 1]
                    for i in range(1, len(sorted_vlm_times))
                ]
                if sorted_vlm_times[0] > 0.0:
                    vlm_gaps.append(sorted_vlm_times[0])
                if bundle.duration_sec > sorted_vlm_times[-1]:
                    vlm_gaps.append(bundle.duration_sec - sorted_vlm_times[-1])
                vlm_max_interval_sec = round(float(max(vlm_gaps)), 4)
            elif len(sorted_vlm_times) == 1:
                vlm_max_interval_sec = round(float(bundle.duration_sec), 4)

            if not vlm_coverage_complete:
                visibility_limitations.append(
                    f"Обзорный проход VLM ограничен {len(executed_overview_windows)} из {len(overview_windows)} окон."
                )

            # Собираем окна для обязательного плотного уточнения (Pass 2 Dense Local Window Refinement, Раздел 6.2, F03)
            refinement_targets: List[Tuple[float, List[CandidateEvent]]] = []

            for c_ev in coarse_confirmed_all:
                tc = float(c_ev["timecode_sec"])
                cand_obj = CandidateEvent(
                    candidate_id=f"VLM_COARSE_{len(all_candidates_log)+1:02d}",
                    code=c_ev["code"],
                    observed_facts=c_ev["observed_facts"],
                    referenced_frame_ids=c_ev["evidence_frame_ids"],
                    estimated_start_sec=max(0.0, tc - 3.0),
                    estimated_end_sec=min(bundle.duration_sec, tc + 3.0),
                    timecode_sec=tc,
                    conditions_to_verify=VIOLATION_CATALOG[c_ev["code"]]["mandatory_visual_indicators"],
                    origin="vlm_coarse",
                )
                all_candidates_log.append(cand_obj)
                refinement_targets.append((tc, [cand_obj]))

            if pipeline_mode == "hybrid":
                for cv_c in enabled_cv_candidates:
                    merged_into_window = False
                    for idx_rt, (t_center, hyps) in enumerate(refinement_targets):
                        same_code = all(h.code == cv_c.code for h in hyps)
                        if (same_code and abs(t_center - cv_c.timecode_sec) <= 4.5) or abs(t_center - cv_c.timecode_sec) <= 1.5:
                            hyps.append(cv_c)
                            new_center = round(sum(h.timecode_sec for h in hyps) / len(hyps), 2)
                            refinement_targets[idx_rt] = (new_center, hyps)
                            merged_into_window = True
                            break
                    if not merged_into_window:
                        refinement_targets.append((cv_c.timecode_sec, [cv_c]))

            # Шаг 4. Плотный уточняющий проход VLM (Pass 2 Dense Local Window) по ВСЕМ кандидатам (F03, Section 6 P2 #8)
            # Ни одно событие из обзорного прохода не публикуется без подтверждения на плотном окне Pass 2,
            # и ни один кандидат не остаётся со статусом 'pending'.
            for win_idx, (center_sec, win_hyps) in enumerate(refinement_targets):
                is_seq_win = any(h.code in SEQUENTIAL_PHASE_CODES for h in win_hyps)
                dense_half_sec = max(float(policy["dense_window_half_sec"]), 8.0 if is_seq_win else 4.5)
                dense_frames = PyAVVideoDecoder.sample_dense_window(
                    bundle=bundle,
                    center_sec=center_sec,
                    window_half_sec=dense_half_sec,
                    max_frames=int(policy["dense_num_frames"]),
                )
                dense_storyboard = PyAVVideoDecoder.build_annotated_storyboard(
                    frames=dense_frames,
                    tile_size=policy["coarse_tile_size"],
                    cols=3,
                )
                dense_stage_label = f"pass2_dense_window_{win_idx+1}_around_{center_sec:.1f}s"
                dense_out, retries_p2, calls_p2 = self._run_vlm_pass_guarded(
                    stage_name=dense_stage_label,
                    storyboard_img=dense_storyboard,
                    shown_frames=dense_frames,
                    duration_sec=bundle.duration_sec,
                    cv_hypotheses=win_hyps,
                    max_tokens=policy["max_tokens"],
                )
                vlm_calls += calls_p2
                vlm_retry_log.extend(retries_p2)
                window_domain_decisions.append(
                    WindowDomainDecisionModel(
                        stage=dense_stage_label,
                        is_banking_cash_room_scene=bool(dense_out.get("is_banking_cash_room_scene", True)),
                        domain_status=str(dense_out.get("domain_status") or "banking_or_workstation_scene"),
                        scene_description=str(dense_out.get("scene_description") or "—"),
                        domain_rejection_reason=dense_out.get("domain_rejection_reason"),
                        shown_frame_ids=[f.frame_id for f in dense_frames],
                    )
                )

                for lim in dense_out.get("visibility_limitations", []):
                    if lim not in visibility_limitations:
                        visibility_limitations.append(lim)

                for dob in dense_out.get("diagnostic_observations", []):
                    code = dob["code"]
                    reason = VIOLATION_CATALOG[code].get("disabled_reason") or "Диагностическое наблюдение"
                    diagnostic_observations.append(
                        DiagnosticObservationModel(
                            code=code,
                            timecode_sec=dob["timecode_sec"],
                            observed_facts=dob["observed_facts"],
                            referenced_frame_ids=dob["referenced_frame_ids"],
                            reason_not_violation=reason,
                        )
                    )

                dense_confirmed = dense_out.get("confirmed_events", [])
                matched_candidate_ids: set = set()

                for dev_idx, dev in enumerate(dense_confirmed):
                    dev_code = dev["code"]
                    dev_tc = float(dev["timecode_sec"])
                    dev_fids_set = set(dev.get("evidence_frame_ids") or [])
                    explicit_cid = dev.get("candidate_id")
                    dev_sig = self._extract_actor_signature(
                        str(dev.get("observed_facts") or ""),
                        str(dev.get("actor_description") or ""),
                    )

                    best_cand: Optional[CandidateEvent] = None
                    if explicit_cid:
                        for h in win_hyps:
                            if h.candidate_id == explicit_cid and h.candidate_id not in matched_candidate_ids:
                                best_cand = h
                                break

                    if best_cand is None:
                        same_code_unmatched = [
                            h for h in win_hyps if h.code == dev_code and h.candidate_id not in matched_candidate_ids
                        ]
                        if same_code_unmatched:
                            def _cand_match_sort_key(cand: CandidateEvent) -> tuple:
                                c_sig = self._extract_actor_signature(
                                    str(cand.observed_facts or ""),
                                    str(cand.candidate_id or ""),
                                )
                                actor_conflict = self._actor_signatures_conflict(dev_sig, c_sig)
                                actor_match_score = len(dev_sig & c_sig)
                                frame_overlap = bool(set(cand.referenced_frame_ids) & dev_fids_set)
                                return (
                                    1 if actor_conflict else 0,
                                    -actor_match_score,
                                    0 if frame_overlap else 1,
                                    abs(float(cand.timecode_sec) - dev_tc),
                                )

                            candidate_ranked = sorted(same_code_unmatched, key=_cand_match_sort_key)
                            top_cand = candidate_ranked[0]
                            top_sig = self._extract_actor_signature(
                                str(top_cand.observed_facts or ""),
                                str(top_cand.candidate_id or ""),
                            )
                            if not self._actor_signatures_conflict(dev_sig, top_sig):
                                best_cand = top_cand

                    if best_cand is not None:
                        matched_candidate_ids.add(best_cand.candidate_id)
                        matching_cand_id = best_cand.candidate_id
                    elif explicit_cid:
                        matching_cand_id = str(explicit_cid)
                    else:
                        matching_cand_id = f"PASS2_WIN_{win_idx+1}_EV_{dev_idx+1}_{dev_code}"

                    primary_fids = list(dict.fromkeys(dev["evidence_frame_ids"]))
                    phased_fids = self._ensure_multiphase_evidence_ids(
                        code=dev_code,
                        primary_frame_ids=primary_fids,
                        window_frames=dense_frames,
                        target_timecode_sec=dev_tc,
                    )
                    structured_conds: Dict[str, Any] = {}
                    for k in (
                        "phone_in_hands",
                        "cash_visible_on_desk",
                        "cashier_left_room",
                        "seal_check_performed",
                        "recount_performed",
                        "safe_door_open",
                        "safe_contents_visible",
                        "client_present",
                        "condition_checks",
                        "mandatory_conditions_confirmed",
                        "episode_id",
                        "actor_description",
                    ):
                        if k in dev and dev[k] is not None:
                            structured_conds[k] = dev[k]

                    confirmed_raw_events.append(
                        {
                            "code": dev_code,
                            "timecode_sec": dev_tc,
                            "observed_facts": dev["observed_facts"],
                            "primary_evidence_frame_ids": primary_fids,
                            "evidence_frame_ids": phased_fids,
                            "candidate_id": matching_cand_id,
                            "episode_id": dev.get("episode_id"),
                            "actor_description": dev.get("actor_description"),
                            "window_idx": win_idx,
                            "structured_conditions": structured_conds,
                            "evidence_source": EVIDENCE_SOURCE_VLM,
                        }
                    )

                self._log_semantic_rejections(dense_out, all_candidates_log, origin="vlm_refined")

                dom_rej_reason = dense_out.get("domain_rejection_reason")
                is_dense_bank = bool(dense_out.get("is_banking_cash_room_scene", True))
                for h in win_hyps:
                    if h.candidate_id in matched_candidate_ids:
                        h.verification_status = "confirmed_violation"
                        h.verification_reason = (
                            "Подтверждено VLM на плотном окне кадров с проверкой обязательных визуальных признаков."
                        )
                    elif not is_dense_bank and dom_rej_reason:
                        h.verification_status = "rejected_by_vlm"
                        h.verification_reason = f"Отклонено по домену сцены: {dom_rej_reason}"
                    else:
                        h.verification_status = "rejected_by_vlm"
                        h.verification_reason = (
                            "Отклонено VLM при плотном покадровом анализе локального окна: обязательные визуальные "
                            "признаки нарушения не подтвердились."
                        )

            vlm_inference_sec = round(time.time() - t_vlm_start, 3)

        else:
            # Режим `cv_only` (для абляционного сравнения Раздела 18.2 без вызова VLM).
            # Ревью 0909, п. 4.1: результат обозначается как непроверенная гипотеза детектора
            # (evidence_source=cv_detector_unverified) без фиктивного подтверждения нормативных условий.
            enabled_cat = get_enabled_catalog()
            for c in all_candidates_log:
                if c.code in enabled_cat:
                    c.verification_status = "confirmed_violation"
                    c.verification_reason = (
                        "Диагностический режим cv_only: гипотеза детектора опубликована без проверки VLM "
                        "(evidence_source=cv_detector_unverified), физические условия правила не подтверждены."
                    )
                    confirmed_raw_events.append(
                        {
                            "code": c.code,
                            "timecode_sec": float(c.timecode_sec),
                            "observed_facts": c.observed_facts,
                            "evidence_frame_ids": c.referenced_frame_ids,
                            "candidate_id": c.candidate_id,
                            "window_idx": 0,
                            "structured_conditions": {},
                            "evidence_source": EVIDENCE_SOURCE_CV,
                        }
                    )
                else:
                    c.verification_status = "diagnostic_only_disabled_rule"
                    c.verification_reason = VIOLATION_CATALOG[c.code].get("disabled_reason") or "Отключённое правило"
                    diagnostic_observations.append(
                        DiagnosticObservationModel(
                            code=c.code,
                            timecode_sec=c.timecode_sec,
                            observed_facts=c.observed_facts,
                            referenced_frame_ids=c.referenced_frame_ids,
                            reason_not_violation=c.verification_reason,
                        )
                    )

        # Шаг 5. Дедупликация одного эпизода при сохранении всех независимых событий (Раздел 9)
        deduped_events = self._merge_same_episode_events(confirmed_raw_events)

        # Шаг 6. Кодирование проверенных доказательных кадров и формирование нормативного вывода
        violations_out: List[Dict[str, Any]] = []
        for ev in deduped_events:
            ev_frames = PyAVVideoDecoder.encode_verified_evidence_frames(
                bundle=bundle,
                frame_ids=ev["evidence_frame_ids"],
                jpeg_quality=88,
            )
            violations_out.append(
                format_violation_entry(
                    code=ev["code"],
                    timecode_sec=ev["timecode_sec"],
                    evidence_frames=ev_frames,
                    custom_observation=ev["observed_facts"],
                    candidate_id=ev.get("candidate_id"),
                    structured_conditions=ev.get("structured_conditions"),
                    evidence_source=ev.get("evidence_source") or EVIDENCE_SOURCE_VLM,
                    episode_id=ev.get("episode_id"),
                    actor_description=ev.get("actor_description"),
                )
            )

        verdict: Literal["violation", "no_violation"] = "violation" if violations_out else "no_violation"
        elapsed_sec = round(time.time() - t_start, 2)

        diagnostics_obj = AnalysisDiagnosticsModel(
            pipeline_mode=pipeline_mode,
            candidates=all_candidates_log,
            diagnostic_observations=diagnostic_observations,
            visibility_limitations=visibility_limitations,
            vlm_retry_log=vlm_retry_log,
            window_domain_decisions=window_domain_decisions,
            analysis_incomplete=(invoke_vlm and not vlm_coverage_complete),
            incomplete_reason=(
                None
                if (not invoke_vlm or vlm_coverage_complete)
                else "Не все обзорные окна длинного видео обработаны в пределах лимита окон."
            ),
        )

        effective_max_interval = (
            max(bundle.max_observation_interval_sec, vlm_max_interval_sec)
            if vlm_max_interval_sec is not None
            else bundle.max_observation_interval_sec
        )

        final_payload: Dict[str, Any] = {
            "verdict": verdict,
            "violations": violations_out,
            "meta": {
                "filename": original_filename or os.path.basename(video_path),
                "duration_sec": round(bundle.duration_sec, 2),
                "resolution": f"{bundle.width}x{bundle.height}",
                "fps": bundle.nominal_fps,
                "codec_name": bundle.codec_name,
                "time_base": bundle.time_base_str,
                "start_offset_sec": bundle.start_offset_sec,
                "offset_normalization_rule": bundle.offset_normalization_rule,
                "is_vfr_detected": bundle.is_vfr_detected,
                "cv_max_observation_interval_sec": bundle.max_observation_interval_sec,
                "vlm_max_observation_interval_sec": vlm_max_interval_sec,
                "max_observation_interval_sec": round(float(effective_max_interval), 4),
                "vlm_coverage_complete": vlm_coverage_complete if invoke_vlm else False,
                "short_action_limitation_note": bundle.short_action_limitation_note,
                "decode_time_sec": decode_elapsed_sec,
                "vlm_inference_time_sec": vlm_inference_sec,
                "processing_time_sec": elapsed_sec,
                "requested_mode": vlm_mode,
                "resolved_mode": resolved_mode,
                "pipeline_mode": pipeline_mode,
                "analysis_policy": policy,
                "vlm_model": APACHE_2_MODEL_ID if invoke_vlm else "cv_only",
                "vlm_revision": APACHE_2_MODEL_REVISION,
                "vlm_license": "Apache-2.0",
                "vlm_calls": vlm_calls,
                "engine_used": f"{pipeline_mode} ({resolved_mode}, {APACHE_2_MODEL_ID})",
                "max_persons_detected": cv_info["max_persons"],
                "yolo_objects_detected": cv_info["detected_interacting_objects"],
                "lights_off_detected": cv_info["lights_off_event"],
                "lights_on_detected": cv_info["lights_on_event"],
                "diagnostics": diagnostics_obj.model_dump(),
            },
        }

        # Шаг 7. Строгая финальная валидация всех инвариантов через Pydantic v2 (Раздел 8.2)
        AnalysisResultModel.model_validate(
            final_payload,
            context={"allowed_frame_ids": allowed_all_fids},
        )

        return final_payload
