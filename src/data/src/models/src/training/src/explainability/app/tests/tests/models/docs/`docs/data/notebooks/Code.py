import os
import sys
import random
import hashlib
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image, UnidentifiedImageError

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

print("All libraries imported successfully.")

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DATASET_PATH = Path("../data/raw/PlantVillage")
PROCESSED_PATH = Path("../data/processed")
REPORTS_PATH = Path("../reports")
MODELS_PATH = Path("../models")

PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
REPORTS_PATH.mkdir(parents=True, exist_ok=True)
MODELS_PATH.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)
print("Dataset:", DATASET_PATH.resolve())

EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

image_paths = [
    p for p in DATASET_PATH.rglob("*")
    if p.is_file()
    and p.suffix.lower() in EXTENSIONS
]

print(f"Total images found: {len(image_paths):,}")

records = []

for path in image_paths:

    class_name = path.parent.name

    if "___" in class_name:
        crop, disease = class_name.split("___", 1)
    else:
        crop = "Unknown"
        disease = class_name

    records.append({
        "image_path": str(path),
        "class": class_name,
        "crop": crop,
        "disease": disease
    })

df = pd.DataFrame(records)

print("Dataset shape:", df.shape)

df.head()

