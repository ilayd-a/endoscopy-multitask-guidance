import cv2
import os
from pathlib import Path

dir = Path("dataset/data/test/images")
output_dir = Path("output")

# First 20 validation images, same ordering as inference script
all_images = sorted([
    f for f in os.listdir(dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])[:20]

for idx, img_name in enumerate(all_images):
    # Load original validation image
    img_path = dir / img_name
    original = cv2.imread(str(img_path))
    original = cv2.resize(original, (256, 256))

    # Load corresponding predicted mask
    mask_path = output_dir / f"{idx:03d}" / "pred_mask.png"
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if original is None:
        print(f"Could not load image: {img_path}")
        continue

    if mask is None:
        print(f"Could not load mask: {mask_path}")
        continue

    # Find contours and draw them in green
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = original.copy()
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    # Save inside each sample folder
    save_path = output_dir / f"{idx:03d}" / "overlay_check.png"
    cv2.imwrite(str(save_path), overlay)

    print(f"Saved {save_path}")