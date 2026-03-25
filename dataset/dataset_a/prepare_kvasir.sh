#!/usr/bin/env bash
# =============================================================================
# prepare_kvasir.sh  –  Download & prepare the Kvasir-SEG dataset
#
# Usage:
#   bash prepare_kvasir.sh
#
# What it does:
#   1. Downloads kvasir-seg.zip from Simula Research Laboratory
#   2. Unzips into ./Kvasir-SEG/
#   3. Verifies the expected 1000 image/mask pairs are present
# =============================================================================

set -euo pipefail

DATASET_URL="https://datasets.simula.no/downloads/kvasir-seg.zip"
ZIP_FILE="kvasir-seg.zip"
EXTRACT_DIR="Kvasir-SEG"
EXPECTED_COUNT=1000

echo "📥 Downloading Kvasir-SEG dataset..."
if command -v wget &>/dev/null; then
    wget -O "$ZIP_FILE" "$DATASET_URL"
elif command -v curl &>/dev/null; then
    curl -k -L -o "$ZIP_FILE" "$DATASET_URL"
else
    echo "❌ Neither wget nor curl found. Please install one and retry."
    exit 1
fi

echo "📦 Extracting..."
unzip -q "$ZIP_FILE" -d "$EXTRACT_DIR"

if [ -d "$EXTRACT_DIR/kvasir-seg" ]; then
    mv "$EXTRACT_DIR/kvasir-seg"/* "$EXTRACT_DIR/"
    rmdir "$EXTRACT_DIR/kvasir-seg"
fi

echo "🔍 Verifying..."
IMG_COUNT=$(find "$EXTRACT_DIR/images" -type f \( -iname "*.jpg" -o -iname "*.png" \) | wc -l)
MASK_COUNT=$(find "$EXTRACT_DIR/masks"  -type f \( -iname "*.jpg" -o -iname "*.png" \) | wc -l)

if [ "$IMG_COUNT" -ne "$EXPECTED_COUNT" ] || [ "$MASK_COUNT" -ne "$EXPECTED_COUNT" ]; then
    echo "⚠️  Warning: expected $EXPECTED_COUNT images and masks."
    echo "   Found $IMG_COUNT images and $MASK_COUNT masks."
else
    echo "✅ Dataset verified: $IMG_COUNT images, $MASK_COUNT masks."
fi

echo ""
echo "🗑️  You can now delete the zip to save space:"
echo "   rm $ZIP_FILE"
echo ""
echo "Next step: open dataset_a.ipynb and run all cells to freeze the split."
