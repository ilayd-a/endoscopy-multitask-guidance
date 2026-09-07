import shutil
import pandas as pd
from pathlib import Path

SOURCE_ROOT = Path("classical_ml/dataset/Kvasir-SEG")
DATA_DIR = Path("classical_ml/dataset/data")
SPLIT_CSV = SOURCE_ROOT / "splits.csv"

df = pd.read_csv(SPLIT_CSV)

for _, row in df.iterrows():
    for kind, col in [("images", "image_path"), ("masks", "mask_path")]:
        src = SOURCE_ROOT / row[col]
        dst = DATA_DIR / row["split"] / kind / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

print("split done")