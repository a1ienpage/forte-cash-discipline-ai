"""
Менеджер очереди и процесс-изолированного выполнения задач с реальным принудительным таймаутом
и атомарным сохранением состояния (Раздел 11 ТЗ на доработку, AT-15, AT-16).

Гарантии эксплуатационной надёжности:
1. Тяжёлый инференс выполняется в отдельном процессе (`multiprocessing` с контекстом `spawn` для совместимости с Apple Metal/MLX).
   Внутри рабочего процесса модель `Qwen2.5-VL-7B-Instruct-4bit` удерживается в памяти между последовательными задачами,
   обеспечивая тёплый старт без перезагрузки весов на каждый запрос.
2. Независимый сторожевой поток (`_supervisor_loop`) контролирует общий срок задачи `deadline_at = accepted_at + timeout_sec`
   каждые 100 мс НЕЗАВИСИМО от вызовов `GET /api/v1/jobs/{job_id}` (AT-15).
3. При истечении `deadline_at` супервизор принудительно завершает зависший процесс (`terminate()` -> `kill()`),
   переводит задачу в терминальный статус `failed`, отвергает любые запоздавшие сообщения по `attempt_token`
   и автоматически поднимает новый процесс-работник для следующей задачи в очереди (AT-15).
4. Все изменения состояния задач записываются атомарно через временный файл и `os.replace()`,
   исключая частично записанный JSON при сбое или перезапуске (AT-16).
"""

import glob
import json
import multiprocessing as mp
import os
import queue
import threading
import time
import traceback
import uuid
from typing import Any, Callable, Dict, Optional

TERMINAL_STATUSES = {"done", "failed"}


def _persistent_worker_main(in_q: mp.Queue, out_q: mp.Queue) -> None:
    """
    Точка входа изолированного рабочего процесса.
    Инициализирует `CashDisciplineAnalyzer` один раз и последовательно обрабатывает задания из `in_q`.
    """
    analyzer = None
    while True:
        task = in_q.get()
        if task is None:
            break

        job_id = task["job_id"]
        attempt_token = task["attempt_token"]
        video_path = task["video_path"]
        original_filename = task["original_filename"]
        vlm_mode = task["vlm_mode"]
        pipeline_mode = task.get("pipeline_mode", "hybrid")
        simulate_sleep_sec = float(task.get("simulate_sleep_sec", 0.0))

        try:
            if simulate_sleep_sec > 0:
                time.sleep(simulate_sleep_sec)

            if analyzer is None:
                from app.analyzer import CashDisciplineAnalyzer

                analyzer = CashDisciplineAnalyzer(default_vlm_mode="auto")

            result = analyzer.analyze_video(
                video_path=video_path,
                original_filename=original_filename,
                vlm_mode=vlm_mode,
                invoke_vlm=(pipeline_mode != "cv_only"),
                pipeline_mode=pipeline_mode,
            )
            out_q.put(
                {
                    "job_id": job_id,
                    "attempt_token": attempt_token,
                    "ok": True,
                    "result": result,
                }
            )
        except Exception as e:
            out_q.put(
                {
                    "job_id": job_id,
                    "attempt_token": attempt_token,
                    "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(limit=5),
                }
            )


class ProcessJobManager:
    """
    Управляет очередью видеозадач, атомарным хранением состояний на диске,
    процесс-работником и независимым контролем таймаутов.
    """

    def __init__(
        self,
        storage_dir: str,
        default_timeout_sec: float = 280.0,
        max_queue_size: int = 32,
        recovery_policy: str = "fail_interrupted",
        readonly_check: Optional[bool] = None,
    ):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.default_timeout_sec = float(default_timeout_sec)
        self.max_queue_size = int(max_queue_size)
        self.recovery_policy = recovery_policy
        self.readonly_check = (
            bool(readonly_check)
            if readonly_check is not None
            else (os.environ.get("FORTE_READONLY_CHECK") == "1")
        )

        self._lock = threading.RLock()
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._pending_queue: "queue.Queue[str]" = queue.Queue(maxsize=self.max_queue_size)

        self._ctx = mp.get_context("spawn")
        self._worker_in_q: Optional[mp.Queue] = None
        self._worker_out_q: Optional[mp.Queue] = None
        self._worker_proc: Optional[mp.Process] = None

        self._stop_event = threading.Event()
        self._custom_runner: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None

        self._supervisor_thread = threading.Thread(
            target=self._supervisor_loop,
            name="JobSupervisorThread",
            daemon=True,
        )
        if not self.readonly_check:
            self.recover_persisted_jobs()
            self._supervisor_thread.start()

    def _job_file_path(self, job_id: str) -> str:
        return os.path.join(self.storage_dir, f"{job_id}.json")

    def _atomic_save_job(self, job_id: str, data: Dict[str, Any]) -> None:
        """
        Атомарно записывает JSON состояния задачи через временный файл и `os.replace`,
        гарантируя отсутствие частично записанных файлов при сбое процесса (Раздел 11.1, AT-16).
        """
        target_path = self._job_file_path(job_id)
        tmp_path = f"{target_path}.tmp.{os.getpid()}.{threading.get_ident()}"
        payload_bytes = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        with open(tmp_path, "wb") as f:
            f.write(payload_bytes)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target_path)

    def recover_persisted_jobs(self) -> int:
        """
        При запуске/перезапуске сервиса проверяет сохранённые на диске задачи.
        Задачи, оставшиеся в нетерминальном состоянии (`processing` / `queued` / `running`),
        переводятся в явное терминальное состояние `failed` с диагностикой перезапуска (Раздел 11.1, AT-16).
        Удаляет осиротевшие `.tmp.*` файлы.
        """
        if self.readonly_check:
            return 0
        recovered_count = 0
        with self._lock:
            for tmp_f in glob.glob(os.path.join(self.storage_dir, "*.json.tmp.*")):
                try:
                    os.remove(tmp_f)
                except OSError:
                    pass

            for path in sorted(glob.glob(os.path.join(self.storage_dir, "*.json"))):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if not isinstance(data, dict) or "job_id" not in data:
                        continue
                    jid = data["job_id"]
                    if data.get("status") not in TERMINAL_STATUSES:
                        data["status"] = "failed"
                        data["internal_state"] = "failed_interrupted_on_restart"
                        data["finished_at"] = time.time()
                        data["error"] = (
                            "Задача была прервана из-за перезапуска сервера (политика восстановления: "
                            "fail_interrupted). Пожалуйста, отправьте запрос повторно."
                        )
                        self._atomic_save_job(jid, data)
                        recovered_count += 1
                    self._jobs[jid] = data
                except Exception:
                    continue
        return recovered_count

    def _ensure_worker_process(self) -> None:
        if self._worker_proc is not None and self._worker_proc.is_alive():
            return
        self._worker_in_q = self._ctx.Queue()
        self._worker_out_q = self._ctx.Queue()
        self._worker_proc = self._ctx.Process(
            target=_persistent_worker_main,
            args=(self._worker_in_q, self._worker_out_q),
            daemon=True,
        )
        self._worker_proc.start()

    def _kill_worker_process(self) -> None:
        proc = self._worker_proc
        self._worker_proc = None
        if proc is not None:
            try:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=0.5)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=0.5)
            except Exception:
                pass

    def submit_job(
        self,
        video_path: str,
        original_filename: str,
        vlm_mode: str = "auto",
        pipeline_mode: str = "hybrid",
        timeout_sec: Optional[float] = None,
        job_id: Optional[str] = None,
        simulate_sleep_sec: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Регистрирует новую задачу в очереди с проверкой лимита очереди и сохранением всех временных меток.
        Предварительно очищает истекшие задачи из `_pending_queue` (R14).
        """
        with self._lock:
            self._sweep_expired_queued_jobs()
            if self._pending_queue.full():
                raise BufferError(
                    f"Очередь задач переполнена (максимум {self.max_queue_size} ожидающих задач). Повторите запрос позже."
                )

            jid = job_id or uuid.uuid4().hex[:10]
            now = time.time()
            eff_timeout = float(timeout_sec if timeout_sec is not None else self.default_timeout_sec)
            attempt_token = uuid.uuid4().hex

            job_record: Dict[str, Any] = {
                "job_id": jid,
                "status": "processing",
                "internal_state": "queued",
                "attempt_token": attempt_token,
                "accepted_at": now,
                "started_at": None,
                "deadline_at": now + eff_timeout,
                "finished_at": None,
                "timeout_sec": eff_timeout,
                "requested_mode": vlm_mode,
                "pipeline_mode": pipeline_mode,
                "filename": original_filename,
                "video_path": video_path,
                "video_url": f"/api/v1/video/{jid}",
                "simulate_sleep_sec": float(simulate_sleep_sec),
                "verdict": None,
                "violations": None,
                "meta": None,
                "error": None,
            }
            self._jobs[jid] = job_record
            self._atomic_save_job(jid, job_record)
            self._pending_queue.put_nowait(jid)
            return dict(job_record)

    def _sweep_expired_queued_jobs(self, now: Optional[float] = None) -> int:
        """
        Немедленно переводит любые ожидающие в очереди задачи с истекшим `deadline_at`
        в терминальный статус `failed` (`failed_queue_timeout`) с атомарной записью на диск (F13)
        и физически удаляет их из `self._pending_queue` для немедленного освобождения слотов очереди (R14).
        """
        curr_now = float(now if now is not None else time.time())
        swept = 0
        with self._lock:
            for q_jid, q_job in list(self._jobs.items()):
                if not isinstance(q_job, dict):
                    continue
                if q_job.get("status") in TERMINAL_STATUSES:
                    continue
                if q_job.get("internal_state") == "queued" and curr_now >= float(q_job.get("deadline_at", curr_now + 1.0)):
                    q_job["status"] = "failed"
                    q_job["internal_state"] = "failed_queue_timeout"
                    q_job["finished_at"] = curr_now
                    q_job["error"] = (
                        f"Задача превысила общий срок ожидания в очереди ({q_job.get('timeout_sec')} сек) до начала расчёта."
                    )
                    self._atomic_save_job(q_jid, q_job)
                    swept += 1

            with self._pending_queue.mutex:
                if self._pending_queue.queue:
                    active_queued = [
                        jid
                        for jid in self._pending_queue.queue
                        if jid in self._jobs
                        and self._jobs[jid].get("status") not in TERMINAL_STATUSES
                        and self._jobs[jid].get("internal_state") == "queued"
                    ]
                    if len(active_queued) != len(self._pending_queue.queue):
                        self._pending_queue.queue.clear()
                        self._pending_queue.queue.extend(active_queued)
                        self._pending_queue.not_full.notify_all()
        return swept

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        self._sweep_expired_queued_jobs()
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                p = self._job_file_path(job_id)
                if os.path.exists(p):
                    try:
                        with open(p, "r", encoding="utf-8") as f:
                            job = json.load(f)
                        self._jobs[job_id] = job
                    except Exception:
                        return None
            return dict(job) if job is not None else None

    def try_apply_worker_message(
        self,
        job_id: str,
        attempt_token: str,
        ok: bool,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> bool:
        """
        Применяет ответ воркера только если:
        1. Задача ещё НЕ находится в терминальном состоянии (`done` / `failed`),
        2. `attempt_token` совпадает с актуальным токеном задачи.
        Запоздавший результат после таймаута (`status == 'failed'`) гарантированно отклоняется (Раздел 11.2, AT-15).
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            if job.get("status") in TERMINAL_STATUSES:
                return False
            if job.get("attempt_token") != attempt_token:
                return False

            now = time.time()
            if now > float(job.get("deadline_at", now + 1.0)):
                job["status"] = "failed"
                job["internal_state"] = "failed_timeout"
                job["finished_at"] = now
                job["error"] = f"Превышен общий лимит времени выполнения задачи ({job.get('timeout_sec')} сек)."
                self._atomic_save_job(job_id, job)
                return False

            if ok and result is not None:
                job["status"] = "done"
                job["internal_state"] = "done"
                job["finished_at"] = now
                job["verdict"] = result["verdict"]
                job["violations"] = result["violations"]
                meta = dict(result.get("meta") or {})
                meta["queue_wait_sec"] = round(
                    max(0.0, float(job.get("started_at") or now) - float(job.get("accepted_at") or now)),
                    3,
                )
                meta["total_wall_time_sec"] = round(
                    max(0.0, now - float(job.get("accepted_at") or now)),
                    3,
                )
                job["meta"] = meta
                job["error"] = None
            else:
                job["status"] = "failed"
                job["internal_state"] = "failed_error"
                job["finished_at"] = now
                job["error"] = error or "Неизвестная ошибка выполнения анализа."

            self._atomic_save_job(job_id, job)
            return True

    def _supervisor_loop(self) -> None:
        """
        Фоновый цикл управления очередью и принудительного контроля сроков выполнения (независимо от GET запросов).
        """
        while not self._stop_event.is_set():
            self._sweep_expired_queued_jobs()
            try:
                jid = self._pending_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            with self._lock:
                self._sweep_expired_queued_jobs()
                job = self._jobs.get(jid)
                if job is None or job.get("status") in TERMINAL_STATUSES:
                    continue
                now = time.time()
                deadline_at = float(job["deadline_at"])
                if now >= deadline_at:
                    job["status"] = "failed"
                    job["internal_state"] = "failed_queue_timeout"
                    job["finished_at"] = now
                    job["error"] = (
                        f"Задача превысила общий срок ожидания в очереди ({job['timeout_sec']} сек) до начала расчёта."
                    )
                    self._atomic_save_job(jid, job)
                    continue

                job["internal_state"] = "running"
                job["started_at"] = now
                self._atomic_save_job(jid, job)
                task_payload = {
                    "job_id": jid,
                    "attempt_token": job["attempt_token"],
                    "video_path": job["video_path"],
                    "original_filename": job["filename"],
                    "vlm_mode": job["requested_mode"],
                    "pipeline_mode": job.get("pipeline_mode", "hybrid"),
                    "simulate_sleep_sec": job.get("simulate_sleep_sec", 0.0),
                }

            # Отправляем задачу в рабочий процесс и ждём с шагом 0.05с, контролируя deadline_at и живучесть процесса
            self._ensure_worker_process()
            assert self._worker_in_q is not None and self._worker_out_q is not None
            self._worker_in_q.put(task_payload)

            finished = False
            while not finished and not self._stop_event.is_set():
                now = time.time()
                self._sweep_expired_queued_jobs(now)
                if now >= deadline_at:
                    # Реальное принудительное завершение зависшего процесса-работника (Раздел 11.2, AT-15)
                    self._kill_worker_process()
                    with self._lock:
                        curr_job = self._jobs.get(jid)
                        if curr_job and curr_job.get("status") not in TERMINAL_STATUSES:
                            curr_job["status"] = "failed"
                            curr_job["internal_state"] = "failed_timeout_killed"
                            curr_job["finished_at"] = now
                            curr_job["error"] = (
                                f"Превышен лимит времени обработки ({curr_job['timeout_sec']} сек). "
                                f"Процесс-работник принудительно остановлен и перезапущен."
                            )
                            self._atomic_save_job(jid, curr_job)
                    finished = True
                    break

                if self._worker_proc is not None and not self._worker_proc.is_alive():
                    # Случай аварийного падения процесса-работника (например, segfault декодера или OOM)
                    try:
                        msg = self._worker_out_q.get_nowait()
                        self.try_apply_worker_message(
                            job_id=msg["job_id"],
                            attempt_token=msg["attempt_token"],
                            ok=msg["ok"],
                            result=msg.get("result"),
                            error=msg.get("error"),
                        )
                    except queue.Empty:
                        with self._lock:
                            curr_job = self._jobs.get(jid)
                            if curr_job and curr_job.get("status") not in TERMINAL_STATUSES:
                                curr_job["status"] = "failed"
                                curr_job["internal_state"] = "failed_worker_crash"
                                curr_job["finished_at"] = now
                                curr_job["error"] = (
                                    f"Процесс-работник аварийно завершился (exitcode={self._worker_proc.exitcode})."
                                )
                                self._atomic_save_job(jid, curr_job)
                    self._kill_worker_process()
                    finished = True
                    break

                try:
                    msg = self._worker_out_q.get(timeout=0.08)
                    if msg.get("job_id") == jid and msg.get("attempt_token") == task_payload["attempt_token"]:
                        self.try_apply_worker_message(
                            job_id=jid,
                            attempt_token=msg["attempt_token"],
                            ok=msg["ok"],
                            result=msg.get("result"),
                            error=msg.get("error"),
                        )
                        finished = True
                except queue.Empty:
                    continue

    def shutdown(self) -> None:
        self._stop_event.set()
        self._kill_worker_process()
