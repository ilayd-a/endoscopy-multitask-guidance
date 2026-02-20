from pathlib import Path
import torch

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR/"data"
MODELS_DIR = PROJECT_DIR/"models"
# The names of your folders with images for each class go here
CLASS_FOLDERS = []
CLASS_DIRS = [DATA_DIR / folder_name for folder_name in CLASS_FOLDERS]

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.mps.is_available():
        return torch.device("mpu")
    else:
        return torch.device("cpu")