# Kvasir-SEG Dataset (Polyp Segmentation)
Need to first download the kvasir-seg.zip dataset from the website https://datasets.simula.no/kvasir-seg/

## Overview
This dataset contains 1,000 polyp images and their corresponding ground truth segmentation masks. It is derived from the Kvasir-SEG dataset (Simula Research Laboratory).

## Directory Structure
The data is split into 80% Train, 10% Validation, and 10% Test.

## Data Format
* **Images:** JPEG/PNG format, RGB color. Various resolutions (approx. 500x500 to 1920x1080).
* **Masks:** Single-channel (Grayscale).
    * **0:** Background (Normal tissue, stool, instruments)
    * **255:** Polyp (The target class)

## Artifacts & Challenges
Be aware of the following when training models:
1.  **Specular Highlights:** Bright white reflections on the mucosal surface. These can be confused with features.
2.  **Instruments:** Some images contain snares or endoscopic tools. The mask correctly labels these as background (0).
3.  **Blur:** Motion blur is present in ~10% of frames due to camera movement.
4.  **Green Box:** Some images may have a green box in the corner (endoscope UI overlay).

## Usage
Use the `KvasirDataset` class in `dataset_a.ipynb` to load this data.
```python
# Example
dataset = KvasirDataset(root_dir="./data", split="train")
image, mask = dataset[0]