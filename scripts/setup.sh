#!/usr/bin/env bash
set -euo pipefail

PYTHON_VERSION="3.11.3"

if ! command -v pyenv >/dev/null 2>&1; then
  echo "pyenv is required but was not found. Install pyenv first." >&2
  exit 1
fi

if ! pyenv versions --bare | grep -q "^${PYTHON_VERSION}$"; then
  echo "Installing Python ${PYTHON_VERSION} with pyenv..."
  pyenv install "${PYTHON_VERSION}"
fi

pyenv local "${PYTHON_VERSION}"

if [ ! -d ".venv" ]; then
  python -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"

echo "Environment setup complete. Activate with: source .venv/bin/activate"
