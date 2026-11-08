#!/usr/bin/env bash
# Script wrapper para execução do script Python de geração do pipeline visual
# Usa o ambiente virtual uv do projeto

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
/home/guilherme/.gemini/antigravity-cli/bin/uv run python scripts/plot_preprocessing_pipeline.py "$@"
