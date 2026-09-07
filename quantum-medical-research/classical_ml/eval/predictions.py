from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from datasets import EvalSample


OUTPUT_ROOT_CANDIDATES = ("outputs", "output")


@dataclass(frozen=True)
class PredictionFiles:
    sample_dir: Path
    pred_mask_path: Path
    pred_heatmap_path: Path
    source_image_name: str | None


def discover_outputs_root(project_root: Path, outputs_dir: Path | None) -> Path:
    if outputs_dir is not None:
        if not outputs_dir.exists():
            raise FileNotFoundError(f"Outputs directory not found: {outputs_dir}")
        return outputs_dir

    for candidate_name in OUTPUT_ROOT_CANDIDATES:
        candidate = project_root / candidate_name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find outputs directory. Checked: {', '.join(OUTPUT_ROOT_CANDIDATES)}"
    )


def discover_prediction_dirs(outputs_root: Path) -> list[PredictionFiles]:
    predictions: list[PredictionFiles] = []
    for sample_dir in sorted(path for path in outputs_root.iterdir() if path.is_dir()):
        pred_mask_path = _find_first_existing(
            sample_dir / "pred_mask.png",
            sample_dir / "pred_mask.npy",
        )
        pred_heatmap_path = _find_first_existing(
            sample_dir / "pred_heatmap.npy",
            sample_dir / "pred_heatmap.png",
        )
        if pred_mask_path is None or pred_heatmap_path is None:
            continue
        predictions.append(
            PredictionFiles(
                sample_dir=sample_dir,
                pred_mask_path=pred_mask_path,
                pred_heatmap_path=pred_heatmap_path,
                source_image_name=_discover_source_image_name(sample_dir),
            )
        )
    if not predictions:
        raise FileNotFoundError(f"No prediction folders found under {outputs_root}")
    return predictions


def match_predictions_to_samples(
    samples: list[EvalSample],
    predictions: list[PredictionFiles],
    allow_index_fallback: bool = False,
) -> list[tuple[EvalSample, PredictionFiles]]:
    key_to_prediction: dict[str, PredictionFiles] = {}
    duplicate_keys: set[str] = set()

    for prediction in predictions:
        for key in _prediction_keys(prediction):
            if key in key_to_prediction:
                duplicate_keys.add(key)
            else:
                key_to_prediction[key] = prediction

    for key in duplicate_keys:
        key_to_prediction.pop(key, None)

    matches: list[tuple[EvalSample, PredictionFiles]] = []
    matched_prediction_dirs: set[Path] = set()
    unmatched_samples: list[EvalSample] = []

    for sample in samples:
        matched_prediction = None
        for key in _sample_keys(sample):
            candidate = key_to_prediction.get(key)
            if candidate is not None and candidate.sample_dir not in matched_prediction_dirs:
                matched_prediction = candidate
                break
        if matched_prediction is None:
            unmatched_samples.append(sample)
            continue
        matched_prediction_dirs.add(matched_prediction.sample_dir)
        matches.append((sample, matched_prediction))

    if unmatched_samples and allow_index_fallback:
        remaining_predictions = [
            prediction
            for prediction in predictions
            if prediction.sample_dir not in matched_prediction_dirs
        ]
        if len(remaining_predictions) == len(unmatched_samples):
            remaining_predictions = sorted(remaining_predictions, key=_numeric_dir_sort_key)
            for sample, prediction in zip(unmatched_samples, remaining_predictions):
                matches.append((sample, prediction))
            unmatched_samples = []

    if unmatched_samples:
        unresolved = ", ".join(sample.sample_id for sample in unmatched_samples[:5])
        raise ValueError(
            "Could not match predictions to dataset samples. "
            "Use exact sample ids, include original_<filename> in each prediction folder, "
            "or pass --allow-index-fallback for legacy numbered outputs. "
            f"Examples: {unresolved}"
        )

    matches.sort(key=lambda pair: pair[0].sample_id)
    return matches


def load_prediction_mask(
    path: Path,
    target_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        mask = np.load(path)
        if mask.ndim == 3:
            mask = mask[..., 0]
    else:
        mask = np.asarray(Image.open(path).convert("L"), dtype=np.float32)

    mask = _resize_array(mask.astype(np.float32), target_shape, resample=Image.NEAREST)
    threshold = 127.0 if mask.max() > 1.0 else 0.5
    return (mask > threshold).astype(np.uint8)


def load_prediction_heatmap(
    path: Path,
    target_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        heatmap = np.load(path).astype(np.float32)
        if heatmap.ndim == 3:
            heatmap = heatmap.mean(axis=-1)
    else:
        heatmap = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
        if heatmap.max() > 0:
            heatmap /= 255.0

    heatmap = _resize_array(heatmap, target_shape, resample=Image.BILINEAR)
    heatmap = np.clip(heatmap, a_min=0.0, a_max=None)
    max_val = float(heatmap.max())
    if max_val > 0.0:
        heatmap /= max_val
    return heatmap.astype(np.float32)


def _find_first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _discover_source_image_name(sample_dir: Path) -> str | None:
    source_file = sample_dir / "source_image.txt"
    if source_file.exists():
        return Path(source_file.read_text(encoding="utf-8").strip()).name

    original_files = sorted(sample_dir.glob("original_*"))
    if original_files:
        return original_files[0].name.removeprefix("original_")

    return None


def _sample_keys(sample: EvalSample) -> set[str]:
    name = sample.image_path.name
    stem = sample.image_path.stem
    return {sample.sample_id, name, stem}


def _prediction_keys(prediction: PredictionFiles) -> set[str]:
    keys = {
        prediction.sample_dir.name,
        prediction.sample_dir.stem,
    }
    if prediction.source_image_name:
        source_name = Path(prediction.source_image_name).name
        keys.add(source_name)
        keys.add(Path(source_name).stem)
    return {key for key in keys if key}


def _numeric_dir_sort_key(prediction: PredictionFiles) -> tuple[int, str]:
    prefix = prediction.sample_dir.name.split("_", 1)[0]
    if prefix.isdigit():
        return int(prefix), prediction.sample_dir.name
    return 10**9, prediction.sample_dir.name


def _resize_array(
    array: np.ndarray,
    target_shape: tuple[int, int] | None,
    resample: int,
) -> np.ndarray:
    array = np.asarray(array)
    if target_shape is None or array.shape == target_shape:
        return array

    image = Image.fromarray(array.astype(np.float32), mode="F")
    image = image.resize((target_shape[1], target_shape[0]), resample)
    return np.asarray(image, dtype=np.float32)
