import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torchvision import models

from app.config import Settings


NUM_CLASSES = 38


@dataclass(frozen=True)
class ModelBundle:
    model: nn.Module
    device: torch.device
    class_names: list[str]


def select_device(device_setting: str) -> torch.device:
    requested = device_setting.lower().strip()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DEVICE=cuda was requested, but CUDA is not available.")
    if requested not in {"cpu", "cuda"}:
        raise RuntimeError("DEVICE must be one of: auto, cpu, cuda.")
    return torch.device(requested)


def load_class_names(path: Path) -> list[str]:
    if not path.exists():
        raise RuntimeError(f"Class names file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        class_names = json.load(file)
    if not isinstance(class_names, list) or not all(isinstance(name, str) for name in class_names):
        raise RuntimeError("Class names file must contain a JSON array of strings.")
    if len(class_names) != NUM_CLASSES:
        raise RuntimeError(f"Expected {NUM_CLASSES} class names, found {len(class_names)}.")
    return class_names


def build_efficientnet_b0(num_classes: int = NUM_CLASSES) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def load_model_bundle(settings: Settings) -> ModelBundle:
    model_path = settings.resolved_model_path
    if not model_path.exists():
        raise RuntimeError(f"Model checkpoint not found: {model_path}")

    class_names = load_class_names(settings.resolved_class_names_path)
    device = select_device(settings.device)
    model = build_efficientnet_b0(num_classes=len(class_names))

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU: N/A")

    return ModelBundle(model=model, device=device, class_names=class_names)
