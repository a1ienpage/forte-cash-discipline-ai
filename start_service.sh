#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PORT=${PORT:-8000}
if [[ -z "${FORTE_API_KEY:-}" && -f "${SCRIPT_DIR}/.forte_api_key" ]]; then
    FORTE_API_KEY="$(tr -d '\r\n' < "${SCRIPT_DIR}/.forte_api_key")"
fi
export FORTE_API_KEY="${FORTE_API_KEY:-}"
export FORTE_JOB_TIMEOUT_SEC=${FORTE_JOB_TIMEOUT_SEC:-280}

RESOLVED_PYTHON=""
for candidate in \
    "${PYTHON_BIN:-}" \
    "${SCRIPT_DIR}/.venv/bin/python3" \
    "${SCRIPT_DIR}/venv/bin/python3" \
    "/opt/miniconda3/bin/python3" \
    "/opt/homebrew/bin/python3" \
    "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3" \
    "/usr/local/bin/python3" \
    "$(command -v python3 2>/dev/null || true)"; do
    if [[ -n "${candidate}" && -x "${candidate}" ]]; then
        if "${candidate}" -c "import uvicorn, fastapi, av, ultralytics" >/dev/null 2>&1; then
            RESOLVED_PYTHON="${candidate}"
            break
        fi
    fi
done

if [[ -z "${RESOLVED_PYTHON}" ]]; then
    echo "ERROR: Не найден интерпретатор Python с установленными зависимостями (uvicorn, fastapi, av, ultralytics)." >&2
    echo "Установите зависимости: python3 -m venv .venv && .venv/bin/pip install -r requirements.lock.txt" >&2
    exit 1
fi

if [[ "${1:-}" == "--check" ]]; then
    FORTE_READONLY_CHECK=1 "${RESOLVED_PYTHON}" -c "import uvicorn, fastapi, av, ultralytics, app.main; print('Environment check OK:', '${RESOLVED_PYTHON}')"
    exit 0
fi

echo "============================================================"
echo " ForteBank Cash Discipline AI Monitor (Datathon'26)"
echo " Working Dir : ${SCRIPT_DIR}"
echo " Python Bin  : ${RESOLVED_PYTHON}"
echo " Local URL   : http://localhost:${PORT}"
echo " X-API-Key   : (set via FORTE_API_KEY env)"
echo " VLM Model   : mlx-community/Qwen2.5-VL-7B-Instruct-4bit (Apache-2.0)"
echo "============================================================"

exec "${RESOLVED_PYTHON}" -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT}"
