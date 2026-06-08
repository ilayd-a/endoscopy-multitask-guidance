#!/usr/bin/env bash
# =============================================================================
# prepare_cvc.sh  –  Prepare the CVC-ClinicDB dataset
#
# Usage:
#   bash prepare_cvc.sh
#
# What it does:
#   1. Looks for archive.zip in the same folder as this script
#   2. Unzips into ./PNG/
#   3. Verifies the expected 612 image/mask pairs are present
#
# Pre-requisites:
#   Manually download the zip from Kaggle first:
#   https://www.kaggle.com/datasets/balraj98/cvcclinicdb
#   Place archive.zip in the same folder as this script, then run:
#   bash prepare_cvc.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ZIP_FILE="archive.zip"
IMAGE_DIR="PNG/Original"
MASK_DIR="PNG/Ground Truth"
EXPECTED_COUNT=612

if [ -d "$IMAGE_DIR" ] && [ -d "$MASK_DIR" ]; then
    echo "ℹ️  PNG/Original and PNG/Ground Truth already exist. Skipping extraction."
else
    if [ ! -f "$ZIP_FILE" ]; then
        echo "❌ archive.zip not found in $(pwd)"
        echo ""
        echo "Please download it manually:"
        echo "  1. Go to: https://www.kaggle.com/datasets/balraj98/cvcclinicdb"
        echo "  2. Click the Download button (top right)"
        echo "  3. Place archive.zip in this folder: $(pwd)"
        echo "  4. Re-run: bash prepare_cvc.sh"
        exit 1
    fi

    echo "📦 Extracting $ZIP_FILE..."
    unzip -q "$ZIP_FILE"
    echo "✅ Extraction complete"
fi

echo "🔍 Verifying..."

if [ ! -d "$IMAGE_DIR" ]; then
    echo "❌ Image folder not found at: $IMAGE_DIR"
    exit 1
fi

if [ ! -d "$MASK_DIR" ]; then
    echo "❌ Mask folder not found at: $MASK_DIR"
    exit 1
fi

IMG_COUNT=$(find "$IMAGE_DIR"  -type f \( -iname "*.jpg" -o -iname "*.png" \) | wc -l)
MASK_COUNT=$(find "$MASK_DIR" -type f \( -iname "*.jpg" -o -iname "*.png" \) | wc -l)

if [ "$IMG_COUNT" -ne "$MASK_COUNT" ]; then
    echo "❌ Mismatch: $IMG_COUNT images vs $MASK_COUNT masks"
    exit 1
fi

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
echo "Next step: open dataset_b.ipynb and run all cells to freeze the split."