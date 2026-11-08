#!/usr/bin/env bash
# Script wrapper para execução do script Python de geração do pipeline visual
# Usa o ambiente virtual uv do projeto

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
# Detecta o comando uv ou python3 disponível no ambiente
# Força Python 3.12 no uv para evitar compilação de pacotes sem wheel (ex: pyarrow em Python >= 3.14)
if command -v uv >/dev/null 2>&1; then
    PY_RUN="uv run --python 3.12 python"
elif [ -f "$HOME/.local/bin/uv" ]; then
    PY_RUN="$HOME/.local/bin/uv run --python 3.12 python"
elif [ -f "$HOME/.cargo/bin/uv" ]; then
    PY_RUN="$HOME/.cargo/bin/uv run --python 3.12 python"
elif [ -f "$HOME/.gemini/antigravity-cli/bin/uv" ]; then
    PY_RUN="$HOME/.gemini/antigravity-cli/bin/uv run --python 3.12 python"
elif command -v python3 >/dev/null 2>&1; then
    PY_RUN="python3"
else
    echo "Erro: Nem 'uv' nem 'python3' foram encontrados no PATH."
    exit 1
fi

$PY_RUN scripts/plot_preprocessing_pipeline.py "$@"
