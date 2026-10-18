"""
FastAPI сервер локального AI-сервиса мониторинга кассовой дисциплины ForteBank (Datathon 26, Версия 2.0).

Реализует:
1. Основной контракт API (Раздел 12 ТЗ на доработку):
   - `POST /api/v1/analyze` (multipart `file`, заголовок `X-API-Key`) -> `{"job_id": "...", "status": "processing"}`
   - `GET  /api/v1/jobs/{job_id}` (заголовок `X-API-Key`) -> статус (`processing`, `done`, `failed`) и результат.
2. Процесс-изолированную очередь задач (`ProcessJobManager` в `app/worker.py`) с:
   - реальным принудительным таймаутом (`JOB_TIMEOUT_SEC`), работающим независимо от `GET`-опросов (AT-15);
   - атомарным сохранением состояний на диск и автоматическим переводом прерванных задач в `failed` при перезапуске (AT-16);
   - защитой от переполнения очереди (`HTTP 429`) и отклонением запоздавших ответов воркера.
3. Безопасную отдачу видеопотока (`GET /api/v1/video/{job_id}`) исключительно по заголовку `X-API-Key`
   (запрещена передача постоянного API-ключа в query-параметре URL для исключения утечки в логи и адресную строку, AT-19).
4. Приём только тех видеоформатов (`.mov`, `.mp4`, `.m4v`), для которых полностью проверена цепочка
   загрузки, PyAV-декодирования, анализа и воспроизведения в браузерном плеере (AT-19).
"""

import os
import shutil
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from app.regulations import (
    AML_DOC,
    AML_URL,
    NBK_231_DOC,
    NBK_231_URL,
    NBK_SECURITY_DOC,
    NBK_SECURITY_URL,
    VIOLATION_CATALOG,
    get_disabled_catalog,
    get_enabled_catalog,
)
from app.video_decoder import PyAVVideoDecoder, VideoDecodeError
from app.vlm_engine import (
    APACHE_2_MODEL_ID,
    APACHE_2_MODEL_REVISION,
    MODE_POLICIES,
    LocalVLMEngine,
)
from app.worker import ProcessJobManager

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
UPLOADS_DIR = os.path.join(BASE_DIR, "jobs_data")
OPEN_VIDEOS_DIR = os.path.join(BASE_DIR, "open", "videos")
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

import secrets

_LEGACY_SOURCE_API_KEY = "forte-datathon26-key"


def _load_configured_api_key() -> str:
    """
    Загружает действующий ключ API из переменной окружения `FORTE_API_KEY` или неверсионируемого файла `.forte_api_key`.
    Встроенный в исходный код ключ не используется как рабочий fallback (ревью 1017, N04).
    """
    env_key = os.environ.get("FORTE_API_KEY", "").strip()
    if env_key and env_key != _LEGACY_SOURCE_API_KEY:
        return env_key
    key_file = os.path.join(BASE_DIR, ".forte_api_key")
    if os.path.isfile(key_file):
        try:
            file_key = open(key_file, "r", encoding="utf-8").read().strip()
            if file_key and file_key != _LEGACY_SOURCE_API_KEY:
                return file_key
        except OSError:
            pass
    return f"ephemeral-{uuid.uuid4().hex}"


API_KEY = _load_configured_api_key()
JOB_TIMEOUT_SEC = float(os.environ.get("FORTE_JOB_TIMEOUT_SEC", "280.0"))
MAX_QUEUE_SIZE = int(os.environ.get("FORTE_MAX_QUEUE_SIZE", "32"))
SUPPORTED_EXTENSIONS = (".mov", ".mp4", ".m4v")

MEDIA_TYPES = {
    ".mov": "video/quicktime",
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
}

job_manager = ProcessJobManager(
    storage_dir=UPLOADS_DIR,
    default_timeout_sec=JOB_TIMEOUT_SEC,
    max_queue_size=MAX_QUEUE_SIZE,
)
_vlm_probe = LocalVLMEngine()


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    job_manager.recover_persisted_jobs()
    yield


app = FastAPI(
    title="ForteBank Cash Discipline AI Monitor",
    description=(
        "Локальный AI-сервис выявления визуально подтверждённых нарушений кассовой "
        "дисциплины по видеозаписей с привязкой к НПА РК (Постановления Правления НБРК № 231 и № 14)."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> str:
    """
    Строгая проверка заголовка `X-API-Key` (Раздел 12 ТЗ, AT-19, N04).
    Передача ключа через query-параметр URL и использование прежнего ключа из исходников запрещены.
    """
    candidate = (x_api_key or "").strip()
    if (
        not candidate
        or candidate == _LEGACY_SOURCE_API_KEY
        or not secrets.compare_digest(candidate, API_KEY)
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing X-API-Key header.",
        )
    return candidate


def _validate_supported_extension(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if not ext or ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Неподдерживаемый формат файла '{ext or 'без расширения'}'. "
                f"Поддерживаются только контейнеры {', '.join(SUPPORTED_EXTENSIONS)} "
                f"с полной проверкой декодирования PyAV и воспроизведения в браузере."
            ),
        )
    return ext


@app.post("/api/v1/analyze", dependencies=[Depends(verify_api_key)])
async def analyze_video_endpoint(
    file: UploadFile = File(...),
    mode: Optional[str] = Form(None),
    vlm_mode: Optional[str] = Form(None),
    pipeline_mode: str = Form("hybrid"),
):
    """
    POST /api/v1/analyze
    Загрузить видео (`.mov`, `.mp4`, `.m4v`, поле `file`).
    Принимает как `vlm_mode`, так и `mode` (`auto` | `fast` | `deep`).
    Возвращает `{"job_id": "...", "status": "processing", "internal_state": "queued"}`.
    Требуется заголовок `X-API-Key`.
    """
    orig_name = file.filename or "uploaded_video.mov"
    ext = _validate_supported_extension(orig_name)

    effective_mode = (vlm_mode or mode or "auto").strip()
    if effective_mode not in ("auto", "fast", "deep"):
        raise HTTPException(
            status_code=400,
            detail=f"Недопустимый режим анализа '{effective_mode}'. Допустимые значения: auto, fast, deep.",
        )
    if pipeline_mode not in ("hybrid", "vlm_only", "cv_only"):
        raise HTTPException(
            status_code=400,
            detail=f"Недопустимый pipeline_mode '{pipeline_mode}'. Допустимые значения: hybrid, vlm_only, cv_only.",
        )

    job_id = uuid.uuid4().hex[:10]
    saved_video_path = os.path.join(UPLOADS_DIR, f"{job_id}{ext}")

    try:
        with open(saved_video_path, "wb") as out_f:
            shutil.copyfileobj(file.file, out_f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка сохранения загруженного файла: {e}") from e

    if os.path.getsize(saved_video_path) == 0:
        try:
            os.remove(saved_video_path)
        except OSError:
            pass
        raise HTTPException(status_code=400, detail="Загруженный видеофайл пуст (0 байт).")

    try:
        job_rec = job_manager.submit_job(
            video_path=saved_video_path,
            original_filename=orig_name,
            vlm_mode=effective_mode,
            pipeline_mode=pipeline_mode,
            job_id=job_id,
        )
    except BufferError as qe:
        raise HTTPException(status_code=429, detail=str(qe)) from qe

    return {
        "job_id": job_rec["job_id"],
        "status": job_rec["status"],
        "internal_state": job_rec.get("internal_state", "queued"),
    }


@app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(verify_api_key)])
async def get_job_status_endpoint(job_id: str):
    """
    GET /api/v1/jobs/{job_id}
    Возвращает статус обработки (`processing`, `done`, `failed`) и полный результат по схеме ТЗ.
    Требуется заголовок `X-API-Key`.
    """
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    status = job.get("status", "processing")
    if status == "processing":
        return {
            "job_id": job_id,
            "status": "processing",
            "internal_state": job.get("internal_state", "running"),
            "accepted_at": job.get("accepted_at"),
            "started_at": job.get("started_at"),
            "deadline_at": job.get("deadline_at"),
        }

    if status == "failed":
        return {
            "job_id": job_id,
            "status": "failed",
            "internal_state": job.get("internal_state", "failed"),
            "error": job.get("error", "Unknown processing error"),
            "accepted_at": job.get("accepted_at"),
            "started_at": job.get("started_at"),
            "finished_at": job.get("finished_at"),
        }

    return {
        "job_id": job["job_id"],
        "status": "done",
        "verdict": job["verdict"],
        "violations": job["violations"],
        "meta": job.get("meta", {}),
        "video_url": job.get("video_url", f"/api/v1/video/{job_id}"),
    }


@app.get("/api/v1/video/{job_id}", dependencies=[Depends(verify_api_key)])
async def stream_job_video(job_id: str):
    """
    Отдаёт видеофайл задачи для воспроизведения в UI через защищённый `fetch()` + `Blob URL`.
    Требуется заголовок `X-API-Key` (передача ключа в URL запрещена по Разделу 12 ТЗ, AT-19).
    """
    job = job_manager.get_job(job_id)
    if job and job.get("video_path") and os.path.exists(job["video_path"]):
        vpath = job["video_path"]
        ext = os.path.splitext(vpath)[1].lower()
        return FileResponse(vpath, media_type=MEDIA_TYPES.get(ext, "video/quicktime"))

    for ext in SUPPORTED_EXTENSIONS:
        candidate = os.path.join(UPLOADS_DIR, f"{job_id}{ext}")
        if os.path.exists(candidate):
            return FileResponse(candidate, media_type=MEDIA_TYPES.get(ext, "video/quicktime"))

    raise HTTPException(status_code=404, detail="Video file not found.")


@app.get("/api/v1/health")
async def health_check():
    """
    GET /api/v1/health
    Техническая проверка фактической работоспособности сервиса, сторожевого потока очереди
    и локального кэша весов 7B VLM (F16, R15).
    """
    vlm_cached = bool(_vlm_probe.is_model_cached())
    supervisor_alive = bool(
        not job_manager._stop_event.is_set()
        and job_manager._supervisor_thread is not None
        and job_manager._supervisor_thread.is_alive()
    )
    queue_depth = int(job_manager._pending_queue.qsize())
    healthy = bool(vlm_cached and supervisor_alive)
    return {
        "status": "ok" if healthy else "degraded",
        "healthy": healthy,
        "supervisor_alive": supervisor_alive,
        "queue_depth": queue_depth,
        "max_queue_size": int(job_manager.max_queue_size),
        "version": "2.0.0",
        "vlm_model_id": APACHE_2_MODEL_ID,
        "vlm_model_cached": vlm_cached,
        "enabled_public_rules_count": len(get_enabled_catalog()),
        "disabled_diagnostic_rules_count": len(get_disabled_catalog()),
        "supported_video_formats": list(SUPPORTED_EXTENSIONS),
    }


@app.get("/api/v1/catalog")
async def get_catalog_and_docs():
    """
    Возвращает перечень проверенных правил кассовой дисциплины (`enabled=True` и `enabled=False`),
    параметры единой локальной 7B-модели и ограничения системы (Разделы 4.2, 10, 12 ТЗ).
    Не раскрывает секретный API-ключ и не возвращает оценочную разметку.
    """
    cached = _vlm_probe.is_model_cached()
    return {
        "vlm_model_info": {
            "model_id": APACHE_2_MODEL_ID,
            "revision": APACHE_2_MODEL_REVISION,
            "license": "Apache-2.0",
            "cached_locally": cached,
            "note": (
                "Во всех режимах (fast, deep, auto) используется одна локальная коммерчески допустимая 7B-модель "
                "Qwen2.5-VL-7B-Instruct-4bit (Apache-2.0). Режимы различаются плотностью первичной раскадровки, "
                "разрешением тайлов и числом уточняющих окон."
            ),
        },
        "vlm_modes": {
            "fast": {
                "model_id": APACHE_2_MODEL_ID,
                "license": "Apache-2.0",
                "cached": cached,
                "policy": MODE_POLICIES["fast"],
            },
            "deep": {
                "model_id": APACHE_2_MODEL_ID,
                "license": "Apache-2.0",
                "cached": cached,
                "policy": MODE_POLICIES["deep"],
            },
        },
        "supported_video_formats": list(SUPPORTED_EXTENSIONS),
        "regulations": [
            {
                "title": NBK_231_DOC,
                "url": NBK_231_URL,
                "scope": "Правила осуществления кассовых операций и операций по инкассации банкнот, монет и ценностей в БВУ РК.",
            },
            {
                "title": NBK_SECURITY_DOC,
                "url": NBK_SECURITY_URL,
                "scope": "Правила организации охраны и устройства помещений банков второго уровня РК.",
            },
            {
                "title": AML_DOC,
                "url": AML_URL,
                "scope": "Требования идентификации клиента (включая сервис цифровых документов по п. 19 пп. 1 Правил № 231).",
            },
        ],
        "enabled_violation_types": list(get_enabled_catalog().values()),
        "disabled_diagnostic_types": list(get_disabled_catalog().values()),
        "violation_types": list(VIOLATION_CATALOG.values()),
        "limitations": [
            "Система анализирует исключительно визуально наблюдаемые действия и состояния (без аудиодорожки и без интеграции с АБИС/СКУД).",
            "Отсутствие объекта или действия в отдельных кадрах не равнозначно невыполнению процедуры вне поля зрения камеры или до начала записи.",
            "Правила, зависящие от внутренних СОП банка (выключение света при уходе, стакан с водой на столе, незакрытый экран ноутбука) или невидимых проводок АБИС, отключены для автоматического обвинения (enabled=False) и фиксируются отдельно как диагностические наблюдения.",
            "Переносимость на новые физические ракурсы камер и новые кассовые процедуры требует отдельной валидации на целевых записях.",
        ],
    }


_dataset_duration_cache: Dict[str, float] = {}


@app.get("/api/v1/dataset", dependencies=[Depends(verify_api_key)])
async def list_dataset_videos():
    """
    Возвращает технический список видеороликов из папки `open/videos/` для тестирования в веб-интерфейсе.
    Требуется заголовок `X-API-Key`. Не возвращает никаких меток или эталонных ответов (Раздел 12 ТЗ).
    """
    if not os.path.isdir(OPEN_VIDEOS_DIR):
        return {"videos": []}
    files = sorted(f for f in os.listdir(OPEN_VIDEOS_DIR) if f.lower().endswith(".mov"))
    items = []
    for fname in files:
        bname = os.path.splitext(fname)[0]
        full_p = os.path.join(OPEN_VIDEOS_DIR, fname)
        dur = _dataset_duration_cache.get(fname)
        if dur is None:
            try:
                meta = PyAVVideoDecoder.probe_container_metadata(full_p)
                dur = round(float(meta.get("duration_sec", 0.0)), 1)
            except Exception:
                dur = 0.0
            _dataset_duration_cache[fname] = dur
        items.append(
            {
                "id": bname,
                "filename": fname,
                "duration_sec": dur,
            }
        )
    return {"videos": items}


@app.post("/api/v1/analyze-sample", dependencies=[Depends(verify_api_key)])
async def analyze_sample_from_dataset(
    video_id: str = Form(...),
    mode: Optional[str] = Form(None),
    vlm_mode: Optional[str] = Form(None),
    pipeline_mode: str = Form("hybrid"),
):
    """
    Запускает анализ выбранного видеоролика из `open/videos/` через общую очередь `ProcessJobManager`.
    Требуется заголовок `X-API-Key`.
    """
    effective_mode = (vlm_mode or mode or "auto").strip()
    if effective_mode not in ("auto", "fast", "deep"):
        raise HTTPException(
            status_code=400,
            detail=f"Недопустимый режим анализа '{effective_mode}'. Допустимые значения: auto, fast, deep.",
        )

    safe_id = os.path.basename(video_id.strip())
    if not safe_id.lower().endswith(".mov"):
        safe_id_file = f"{safe_id}.mov"
    else:
        safe_id_file = safe_id

    src_path = os.path.join(OPEN_VIDEOS_DIR, safe_id_file)
    if not os.path.exists(src_path):
        raise HTTPException(status_code=404, detail=f"Видео '{safe_id_file}' не найдено в open/videos/.")

    job_id = uuid.uuid4().hex[:10]
    dst_path = os.path.join(UPLOADS_DIR, f"{job_id}.mov")
    shutil.copy2(src_path, dst_path)

    try:
        job_rec = job_manager.submit_job(
            video_path=dst_path,
            original_filename=safe_id_file,
            vlm_mode=effective_mode,
            pipeline_mode=pipeline_mode,
            job_id=job_id,
        )
    except BufferError as qe:
        raise HTTPException(status_code=429, detail=str(qe)) from qe

    return {
        "job_id": job_rec["job_id"],
        "status": job_rec["status"],
        "internal_state": job_rec.get("internal_state", "queued"),
    }


@app.get("/", response_class=HTMLResponse)
async def root_ui():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse("<h1>ForteBank Cash Discipline AI Service Running</h1>")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())
