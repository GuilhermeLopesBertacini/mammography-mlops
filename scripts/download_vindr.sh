#!/usr/bin/env bash
set -euo pipefail

# Script para download do dataset VinDr-Mammo (Arquivo ZIP Oficial: 50.7 GB)
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

ZIP_FILENAME="vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0.zip"
ZIP_URL="${PHYSIONET_VINDR_ZIP_URL:-https://physionet.org/static/published-projects/vindr-mammo/$ZIP_FILENAME}"

echo "=========================================================="
echo "Download Oficial do VinDr-Mammo (Arquivo ZIP - 50.7 GB)"
echo "Destino: $DEST_DIR"
echo "Usuário: $PHYSIONET_USER"
echo "=========================================================="

cd "$DEST_DIR"

if [ -n "${PHYSIONET_PASSWORD:-}" ]; then
    wget -c \
        --user="$PHYSIONET_USER" \
        --password="$PHYSIONET_PASSWORD" \
        "$ZIP_URL"
else
    echo "PHYSIONET_PASSWORD não preenchido no .env. Solicitando senha interativamente:"
    wget -c \
        --user="$PHYSIONET_USER" \
        --ask-password \
        "$ZIP_URL"
fi

echo ""
echo "Download do arquivo ZIP concluído com sucesso em $DEST_DIR/$ZIP_FILENAME"
echo "Para descompactar (expandirá para ~337 GB), execute:"
echo "  unzip -q $DEST_DIR/$ZIP_FILENAME -d $DEST_DIR"
