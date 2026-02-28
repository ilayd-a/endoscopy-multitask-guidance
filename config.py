from pathlib import Path
import torch

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR/"dataset"
MODELS_DIR = PROJECT_DIR/"models"

# The names of your folders with images for each class go here
CLASS_FOLDERS = []
CLASS_DIRS = [DATA_DIR / folder_name for folder_name in CLASS_FOLDERS]

OUTPUT_DIR = PROJECT_DIR/"output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")