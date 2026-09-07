from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CVC_ROOT = PROJECT_ROOT / "dataset" / "CVC-ClinicDB"
DEFAULT_SPLIT_ROOT = PROJECT_ROOT / "dataset" / "data"
DEFAULT_KVASIR_ROOT = PROJECT_ROOT / "dataset" / "Kvasir-SEG"


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
    split_manifest: Path | None = None,
    split_column: str | None = None,
) -> list[EvalSample]:
    if dataset == "cvc":
        root = dataset_root or DEFAULT_CVC_ROOT
        return load_cvc_samples(
            root,
            split,
            metadata_path=metadata_path,
            split_manifest=split_manifest,
            split_column=split_column,
        )
    if dataset == "kvasir":
        root = dataset_root or (
            DEFAULT_KVASIR_ROOT
            if (DEFAULT_KVASIR_ROOT / "splits.csv").exists()
            else DEFAULT_SPLIT_ROOT
        )
        return load_kvasir_samples(root, split)
    if dataset == "split_csv":
        if dataset_root is None:
            raise ValueError("--dataset-root is required when --dataset split_csv is used")
        return load_split_csv_samples(dataset_root, split)
    if dataset == "split_folder":
        root = dataset_root or DEFAULT_SPLIT_ROOT
        return load_split_folder_samples(root, split)
    raise ValueError(f"Unsupported dataset: {dataset}")


def load_cvc_samples(
    dataset_root: Path,
    split: str,
    metadata_path: Path | None = None,
    split_manifest: Path | None = None,
    split_column: str | None = None,
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
    if "png_image_path" not in metadata.columns:
        raise ValueError(
            "CVC metadata.csv must contain a 'png_image_path' column"
        )

    split_df = resolve_cvc_split(
        metadata=metadata,
        split=split,
        dataset_root=dataset_root,
        split_manifest=split_manifest,
        split_column=split_column,
    )
    split_df = sort_split_df(split_df)
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


def resolve_cvc_split(
    metadata: pd.DataFrame,
    split: str,
    dataset_root: Path,
    split_manifest: Path | None = None,
    split_column: str | None = None,
) -> pd.DataFrame:
    manifest_path = split_manifest or discover_split_manifest(dataset_root, split)
    if manifest_path is not None:
        return filter_metadata_by_manifest(metadata, manifest_path)

    if split_column is not None:
        if split_column not in metadata.columns:
            raise ValueError(
                f"Requested split column '{split_column}' was not found in metadata.csv"
            )
        return filter_metadata_by_split_column(metadata, split_column, split)

    auto_column = discover_split_column(metadata, split)
    if auto_column is not None:
        return filter_metadata_by_split_column(metadata, auto_column, split)

    raise ValueError(
        "CVC evaluation requires an official split artifact from the data team. "
        "Pass --split-manifest, or add a split column to metadata.csv, or place "
        "shared files such as splits/train.txt, splits/val.txt, and splits/test.txt "
        "under the dataset root."
    )


def discover_split_manifest(dataset_root: Path, split: str) -> Path | None:
    candidates = [
        dataset_root / "splits" / f"{split}.txt",
        dataset_root / "splits" / f"{split}.csv",
        dataset_root / f"{split}.txt",
        dataset_root / f"{split}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def discover_split_column(metadata: pd.DataFrame, split: str) -> str | None:
    preferred_columns = ("split", "official_split", "dataset_split")
    for column in preferred_columns:
        if column in metadata.columns:
            return column

    if split in metadata.columns:
        return split

    prefixed_column = f"is_{split}"
    if prefixed_column in metadata.columns:
        return prefixed_column

    return None


def filter_metadata_by_manifest(metadata: pd.DataFrame, manifest_path: Path) -> pd.DataFrame:
    allowed_names = load_manifest_image_names(manifest_path)
    image_names = metadata["png_image_path"].map(lambda value: Path(value).name)
    split_df = metadata.loc[image_names.isin(allowed_names)].copy()
    if split_df.empty:
        raise ValueError(
            f"Split manifest matched zero samples from metadata.csv: {manifest_path}"
        )
    return split_df


def load_manifest_image_names(manifest_path: Path) -> set[str]:
    if manifest_path.suffix.lower() == ".txt":
        rows = manifest_path.read_text(encoding="utf-8").splitlines()
        names = {Path(row.strip()).name for row in rows if row.strip()}
    elif manifest_path.suffix.lower() == ".csv":
        manifest_df = pd.read_csv(manifest_path)
        candidates = ("png_image_path", "image_name", "sample_id", "filename")
        column = next((name for name in candidates if name in manifest_df.columns), None)
        if column is None:
            raise ValueError(
                "Split CSV must contain one of: png_image_path, image_name, sample_id, filename"
            )
        names = {
            Path(str(value)).name
            for value in manifest_df[column].dropna().tolist()
            if str(value).strip()
        }
    else:
        raise ValueError(f"Unsupported split manifest format: {manifest_path}")

    if not names:
        raise ValueError(f"Split manifest is empty: {manifest_path}")
    return names


def filter_metadata_by_split_column(
    metadata: pd.DataFrame,
    split_column: str,
    split: str,
) -> pd.DataFrame:
    series = metadata[split_column]

    if pd.api.types.is_bool_dtype(series):
        split_df = metadata.loc[series.fillna(False)].copy()
    elif split_column in {split, f"is_{split}"}:
        numeric = pd.to_numeric(series, errors="coerce")
        split_df = metadata.loc[numeric.fillna(0).astype(int) == 1].copy()
    else:
        normalized = series.astype(str).str.strip().str.lower()
        split_df = metadata.loc[normalized == split.lower()].copy()

    if split_df.empty:
        raise ValueError(
            f"Split column '{split_column}' produced zero samples for split '{split}'"
        )
    return split_df


def sort_split_df(split_df: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [
        column for column in ("sequence_id", "frame_id", "png_image_path")
        if column in split_df.columns
    ]
    if sort_columns:
        return split_df.sort_values(sort_columns)
    return split_df.sort_values("png_image_path")


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


def load_kvasir_samples(dataset_root: Path, split: str) -> list[EvalSample]:
    for candidate_root in (dataset_root, dataset_root / "Kvasir-SEG"):
        if (candidate_root / "splits.csv").exists():
            return load_split_csv_samples(candidate_root, split)
    return load_split_folder_samples(dataset_root, split)


def load_split_csv_samples(dataset_root: Path, split: str) -> list[EvalSample]:
    split_csv_path = dataset_root / "splits.csv"
    if not split_csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_csv_path}")

    frame = pd.read_csv(split_csv_path)
    if "split" not in frame.columns:
        raise ValueError(f"{split_csv_path} must contain a 'split' column")

    image_column = next(
        (
            column
            for column in ("image_path", "image", "image_file", "png_image_path")
            if column in frame.columns
        ),
        None,
    )
    mask_column = next(
        (
            column
            for column in ("mask_path", "mask", "mask_file")
            if column in frame.columns
        ),
        None,
    )
    if image_column is None or mask_column is None:
        raise ValueError(
            f"{split_csv_path} must contain image and mask path columns"
        )

    split_df = frame.loc[
        frame["split"].astype(str).str.strip().str.lower() == split.lower()
    ].copy()
    if split_df.empty:
        raise ValueError(f"{split_csv_path} contains no rows for split '{split}'")

    sort_columns = [
        column for column in ("stem", image_column, mask_column) if column in split_df.columns
    ]
    split_df = split_df.sort_values(sort_columns)

    samples: list[EvalSample] = []
    for _, row in split_df.iterrows():
        image_path = dataset_root / str(row[image_column])
        mask_path = dataset_root / str(row[mask_column])
        if not image_path.exists() or not mask_path.exists():
            raise FileNotFoundError(
                f"Missing image/mask pair from split CSV: {image_path.name}"
            )
        sample_id = str(row["stem"]) if "stem" in row and pd.notna(row["stem"]) else image_path.name
        samples.append(
            EvalSample(
                sample_id=sample_id,
                image_path=image_path,
                mask_path=mask_path,
            )
        )
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
