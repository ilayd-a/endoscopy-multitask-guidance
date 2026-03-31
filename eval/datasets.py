from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CVC_ROOT = PROJECT_ROOT / "dataset" / "CVC-ClinicDB"
DEFAULT_SPLIT_ROOT = PROJECT_ROOT / "dataset" / "data"


@dataclass(frozen=True)
class EvalSample:
    sample_id: str
    image_path: Path
    mask_path: Path


def load_samples(
    dataset: str,
    split: str,
    dataset_root: Path | None = None,
    metadata_path: Path | None = None,
) -> list[EvalSample]:
    if dataset == "cvc":
        root = dataset_root or DEFAULT_CVC_ROOT
        return load_cvc_samples(root, split, metadata_path)
    if dataset in {"kvasir", "split_folder"}:
        root = dataset_root or DEFAULT_SPLIT_ROOT
        return load_split_folder_samples(root, split)
    raise ValueError(f"Unsupported dataset: {dataset}")


def load_cvc_samples(
    dataset_root: Path,
    split: str,
    metadata_path: Path | None = None,
) -> list[EvalSample]:
    metadata_path = metadata_path or dataset_root / "metadata.csv"
    image_dir = dataset_root / "PNG" / "Original"
    mask_dir = dataset_root / "PNG" / "Ground Truth"

    if not metadata_path.exists():
        raise FileNotFoundError(f"CVC metadata not found: {metadata_path}")
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(
            f"CVC image/mask folders not found under {dataset_root}"
        )

    metadata = pd.read_csv(metadata_path)
    if "sequence_id" not in metadata.columns or "png_image_path" not in metadata.columns:
        raise ValueError(
            "CVC metadata.csv must contain 'sequence_id' and 'png_image_path' columns"
        )

    if split == "train":
        split_df = metadata[metadata["sequence_id"] <= 23]
    elif split == "val":
        split_df = metadata[(metadata["sequence_id"] >= 24) & (metadata["sequence_id"] <= 26)]
    elif split == "test":
        split_df = metadata[metadata["sequence_id"] >= 27]
    else:
        raise ValueError(f"Unsupported split for CVC: {split}")

    split_df = split_df.sort_values(["sequence_id", "png_image_path"])
    samples: list[EvalSample] = []
    for image_rel in split_df["png_image_path"].tolist():
        image_name = Path(image_rel).name
        image_path = image_dir / image_name
        mask_path = mask_dir / image_name
        if not image_path.exists() or not mask_path.exists():
            raise FileNotFoundError(
                f"Missing CVC image/mask pair for {image_name}"
            )
        samples.append(EvalSample(sample_id=image_name, image_path=image_path, mask_path=mask_path))
    return samples


def load_split_folder_samples(dataset_root: Path, split: str) -> list[EvalSample]:
    image_dir = dataset_root / split / "images"
    mask_dir = dataset_root / split / "masks"
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(
            f"Expected split folder layout at {dataset_root / split}"
        )

    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )
    samples: list[EvalSample] = []
    for image_path in image_paths:
        mask_path = mask_dir / image_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {image_path.name}")
        samples.append(EvalSample(sample_id=image_path.name, image_path=image_path, mask_path=mask_path))
    return samples


def load_rgb_image(path: Path, target_shape: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if target_shape is not None:
        image = image.resize((target_shape[1], target_shape[0]), Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def load_binary_mask(path: Path, target_shape: tuple[int, int] | None = None) -> np.ndarray:
    mask = Image.open(path).convert("L")
    if target_shape is not None:
        mask = mask.resize((target_shape[1], target_shape[0]), Image.NEAREST)
    mask_array = np.asarray(mask, dtype=np.float32)
    threshold = 127.0 if mask_array.max() > 1.0 else 0.5
    return (mask_array > threshold).astype(np.uint8)
