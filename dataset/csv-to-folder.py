import pandas as pd
import shutil
from pathlib import Path

SOURCE_ROOT = Path("Kvasir-SEG")
DEST_ROOT   = Path("data")
df = pd.read_csv(SOURCE_ROOT / "splits.csv")

for _, row in df.iterrows():
    for kind, col in [("images", "image_path"), ("masks", "mask_path")]:
        src  = SOURCE_ROOT / row[col]
        dest = DEST_ROOT / row["split"] / kind / src.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

