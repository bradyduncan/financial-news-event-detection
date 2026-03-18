#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${PROJECT_ROOT}/.venv"
REQ_PATH="${PROJECT_ROOT}/requirements.txt"

if [ ! -d "${VENV_PATH}" ]; then
  python3 -m venv "${VENV_PATH}"
fi

ACTIVATE="${VENV_PATH}/bin/activate"
if [ ! -f "${ACTIVATE}" ]; then
  echo "Virtual environment activation script not found: ${ACTIVATE}" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${ACTIVATE}"
python -m pip install --upgrade pip
python -m pip install -r "${REQ_PATH}"

echo "Venv ready: ${VENV_PATH}"
echo "Activate with: source ${ACTIVATE}"
