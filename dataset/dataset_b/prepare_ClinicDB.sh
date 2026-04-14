#!/usr/bin/env bash
# =============================================================================
# prepare_ClinicDB.sh  –  Download & prepare the CVC-ClinicDB dataset
#
# Usage:
#   bash prepare_ClinicDB.sh
#
# What it does:
#   1. Downloads cvcclinicdb via the Kaggle API
#   2. Unzips into ./archive/ (aligning with dataset_b.ipynb SOURCE_ROOT)
#   3. Verifies the expected 612 image/mask pairs are present
# =============================================================================

set -euo pipefail

DATASET="balraj98/cvcclinicdb"
ZIP_FILE="cvcclinicdb.zip"
EXTRACT_DIR="archive"
EXPECTED_COUNT=612

echo "📥 Checking for Kaggle CLI..."
if ! command -v kaggle &>/dev/null; then
    echo "❌ Kaggle CLI is not installed."
    echo "   Please run 'pip install kaggle' and ensure your kaggle.json is configured."
    exit 1
fi

echo "📥 Downloading CVC-ClinicDB dataset from Kaggle..."
# Download the dataset into the current directory
kaggle datasets download -d "$DATASET" -p .

echo "📦 Extracting into $EXTRACT_DIR/..."
mkdir -p "$EXTRACT_DIR"
# The Kaggle zip contains PNG/ and TIF/ folders at the root.
# Extracting into archive/ creates archive/PNG/Original and archive/PNG/Ground Truth
unzip -q "$ZIP_FILE" -d "$EXTRACT_DIR"

echo "🔍 Verifying..."
# The target notebook dataset_b.ipynb specifically looks at the PNG folder
IMG_COUNT=$(find "$EXTRACT_DIR/PNG/Original" -type f \( -iname "*.jpg" -o -iname "*.png" \) | wc -l)
MASK_COUNT=$(find "$EXTRACT_DIR/PNG/Ground Truth" -type f \( -iname "*.jpg" -o -iname "*.png" \) | wc -l)

if [ "$IMG_COUNT" -ne "$EXPECTED_COUNT" ] || [ "$MASK_COUNT" -ne "$EXPECTED_COUNT" ]; then
    echo "⚠️  Warning: expected $EXPECTED_COUNT images and masks."
    echo "   Found $IMG_COUNT images and $MASK_COUNT masks."
else
    echo "✅ Dataset verified: $IMG_COUNT images, $MASK_COUNT masks ready for segmentation."
fi

echo ""
echo "🗑️  You can now delete the zip file to save space:"
echo "   rm $ZIP_FILE"
echo ""
echo "Next step: open dataset_b.ipynb and run all cells to freeze the split."