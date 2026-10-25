#!/usr/bin/env bash
set -euo pipefail

# Script para download do dataset VinDr-Mammo via PhysioNet
# Carrega credenciais do arquivo .env

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_ROOT/.env"

if [ -f "$ENV_FILE" ]; then
    # Exporta variáveis do .env ignorando comentários
    export $(grep -v '^#' "$ENV_FILE" | xargs)
else
    echo "Erro: Arquivo .env não encontrado em $ENV_FILE"
    echo "Copie .env.example para .env e preencha suas credenciais."
    exit 1
fi

if [ -z "${PHYSIONET_USER:-}" ]; then
    echo "Erro: PHYSIONET_USER não definido no .env"
    exit 1
fi

DEST_DIR="${1:-$PROJECT_ROOT/data/raw/vindr-mammo}"
mkdir -p "$DEST_DIR"

URL="${PHYSIONET_VINDR_URL:-https://physionet.org/files/vindr-mammo/1.0.0/}"

echo "Iniciando download do VinDr-Mammo para: $DEST_DIR"
echo "Usuário PhysioNet: $PHYSIONET_USER"

if [ -n "${PHYSIONET_PASSWORD:-}" ]; then
    wget -r -N -c -np \
        --user="$PHYSIONET_USER" \
        --password="$PHYSIONET_PASSWORD" \
        -P "$DEST_DIR" \
        "$URL"
else
    echo "PHYSIONET_PASSWORD não preenchido no .env. Solicitando senha interativamente:"
    wget -r -N -c -np \
        --user="$PHYSIONET_USER" \
        --ask-password \
        -P "$DEST_DIR" \
        "$URL"
fi

echo "Download concluído."
