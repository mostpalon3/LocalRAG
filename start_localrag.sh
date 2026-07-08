#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"
STREAMLIT_BIN="$VENV_DIR/bin/streamlit"

if ! command -v ollama >/dev/null 2>&1; then
  echo "ollama is not installed or not on PATH. Install Ollama first, then rerun this script." >&2
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install -r "$ROOT_DIR/requirements.txt"

ollama pull qwen3:8b
ollama pull nomic-embed-text

echo "Starting Streamlit..."
exec "$STREAMLIT_BIN" run "$ROOT_DIR/streamlit_app.py"