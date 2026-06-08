#!/usr/bin/env bash
#=============================================================================
#prepare_kvasir.sh  –  Download & prepare the Kvasir-SEG dataset
#=============================================================================
set -euo pipefail

#✅ Ensure script always runs from the repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DATASET_URL="https://datasets.simula.no/downloads/kvasir-seg.zip"
ZIP_FILE="dataset/dataset_a/kvasir-seg.zip"
RAW_DIR="dataset/dataset_a/Kvasir-SEG"
TARGET_DIR="dataset/data"
EXPECTED_COUNT=1000
TRAIN_COUNT=800
VAL_COUNT=100
TEST_COUNT=100

echo "📥 Downloading Kvasir-SEG dataset..."
if command -v wget &>/dev/null; then
    wget -O "$ZIP_FILE" "$DATASET_URL"
elif command -v curl &>/dev/null; then
    curl -k -L -o "$ZIP_FILE" "$DATASET_URL"
else
    echo "Neither wget nor curl found. Please install one and retry."
    exit 1
fi

echo "📦 Extracting..."
rm -rf "$RAW_DIR" "$TARGET_DIR"
mkdir -p "$RAW_DIR"
unzip -q "$ZIP_FILE" -d "$RAW_DIR"

if [ -d "$RAW_DIR/kvasir-seg" ]; then
    mv "$RAW_DIR/kvasir-seg"/* "$RAW_DIR/"
    rmdir "$RAW_DIR/kvasir-seg"
fi

echo "🔍 Verifying..."
IMG_COUNT=$(find "$RAW_DIR/images" -type f \( -iname "*.jpg" -o -iname "*.png" \) | wc -l)
MASK_COUNT=$(find "$RAW_DIR/masks"  -type f \( -iname "*.jpg" -o -iname "*.png" \) | wc -l)

if [ "$IMG_COUNT" -ne "$EXPECTED_COUNT" ] || [ "$MASK_COUNT" -ne "$EXPECTED_COUNT" ]; then
    echo "⚠️  Warning: expected $EXPECTED_COUNT images and masks."
    echo "   Found $IMG_COUNT images and $MASK_COUNT masks."
else
    echo "✅ Dataset verified: $IMG_COUNT images, $MASK_COUNT masks."
fi

echo "🧊 Creating deterministic train/val/test split in $TARGET_DIR..."
mkdir -p \
    "$TARGET_DIR/train/images" "$TARGET_DIR/train/masks" \
    "$TARGET_DIR/val/images" "$TARGET_DIR/val/masks" \
    "$TARGET_DIR/test/images" "$TARGET_DIR/test/masks"

IMAGES=()
while IFS= read -r image_path; do
    IMAGES+=("$image_path")
done < <(find "$RAW_DIR/images" -type f \( -iname "*.jpg" -o -iname "*.png" \) | sort)

copy_split() {
    local split_name="$1"
    local start="$2"
    local count="$3"
    local end=$((start + count))

    for ((i = start; i < end; i++)); do
        image_path="${IMAGES[$i]}"
        image_name="$(basename "$image_path")"
        mask_path="$RAW_DIR/masks/$image_name"

        if [ ! -f "$mask_path" ]; then
            echo "Missing mask for $image_name"
            exit 1
        fi

        cp "$image_path" "$TARGET_DIR/$split_name/images/$image_name"
        cp "$mask_path" "$TARGET_DIR/$split_name/masks/$image_name"
    done
}

copy_split train 0 "$TRAIN_COUNT"
copy_split val "$TRAIN_COUNT" "$VAL_COUNT"
copy_split test $((TRAIN_COUNT + VAL_COUNT)) "$TEST_COUNT"

{
    echo "split,image_path,mask_path,stem"
    for split_name in train val test; do
        for image_path in "$TARGET_DIR/$split_name/images"/*; do
            image_name="$(basename "$image_path")"
            stem="${image_name%.*}"
            echo "$split_name,$split_name/images/$image_name,$split_name/masks/$image_name,$stem"
        done
    done
} > "$TARGET_DIR/splits.csv"

echo "✅ Prepared split folder used by config.py and eval:"
echo "   $TARGET_DIR/train: $TRAIN_COUNT images"
echo "   $TARGET_DIR/val:   $VAL_COUNT images"
echo "   $TARGET_DIR/test:  $TEST_COUNT images"
echo "   $TARGET_DIR/splits.csv"

echo ""
echo "🗑️  You can now delete the zip to save space:"
echo "   rm $ZIP_FILE"
echo ""
echo "Next steps:"
echo "   PYTHONPATH=. python3 models/train_pretrained.py"
echo "   PYTHONPATH=. python3 models/heatmap_pretrained.py"
echo "   python3 eval/evaluate_outputs.py --dataset kvasir --split val --outputs-dir output --run-name pretrained --allow-index-fallback"
