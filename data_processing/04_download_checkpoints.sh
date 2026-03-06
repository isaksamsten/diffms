#!/usr/bin/env bash
# Download pretrained DiffMS checkpoints from Zenodo.
#
# The Zenodo archive contains:
#   - diffms_checkpoints.tar.gz  (encoder, decoder, and finetuned model weights)
#   - msg_preprocessed.tar.gz    (preprocessed MassSpecGym data)
#
# Usage:
#   bash data_processing/04_download_checkpoints.sh
#
# After downloading, set the paths in configs/general/general_default.yaml.

set -euo pipefail

ZENODO_URL="https://zenodo.org/api/records/15122968/files-archive"
CKPT_DIR="checkpoints"
ARCHIVE="zenodo_archive.zip"

mkdir -p "$CKPT_DIR"

echo "=== Downloading DiffMS files from Zenodo ==="
echo "    Source: https://zenodo.org/records/15122968"
echo ""

if [ -f "$ARCHIVE" ]; then
    echo "[skip] $ARCHIVE already exists"
else
    echo "[download] Zenodo files archive ..."
    wget -q --show-progress -O "$ARCHIVE" "$ZENODO_URL"
fi

echo "[extract] Unpacking Zenodo archive ..."
unzip -o -q "$ARCHIVE" -d "$CKPT_DIR"

if [ -f "$CKPT_DIR/diffms_checkpoints.tar.gz" ]; then
    echo "[extract] diffms_checkpoints.tar.gz ..."
    tar -xzf "$CKPT_DIR/diffms_checkpoints.tar.gz" -C "$CKPT_DIR"
    rm -f "$CKPT_DIR/diffms_checkpoints.tar.gz"
else
    echo "[warn] diffms_checkpoints.tar.gz not found in archive"
fi

if [ -f "$CKPT_DIR/msg_preprocessed.tar.gz" ]; then
    echo "[extract] msg_preprocessed.tar.gz -> data/msg/ ..."
    mkdir -p data/msg
    tar -xzf "$CKPT_DIR/msg_preprocessed.tar.gz" -C data/msg
    mv "$CKPT_DIR/msg_preprocessed.tar.gz" data/msg/
fi

rm -f "$ARCHIVE"

echo "Checkpoint files in ${CKPT_DIR}/:"
ls -lh "$CKPT_DIR"/ 2>/dev/null || echo "  (empty)"
