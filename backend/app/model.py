import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torchvision import models

from app.config import Settings


NUM_CLASSES = 38
EXPECTED_CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry___Powdery_mildew",
    "Cherry___healthy",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


@dataclass(frozen=True)
class ModelBundle:
    model: nn.Module
    device: torch.device
    class_names: list[str]
    model_version: str
    model_path: Path


def select_device(device_setting: str) -> torch.device:
    requested = device_setting.lower().strip()

    if requested == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "DEVICE=cuda was requested, but CUDA is not available."
            )
        return torch.device("cuda")

    if requested == "cpu":
        return torch.device("cpu")

    raise RuntimeError(
        "DEVICE must be one of: auto, cpu, cuda."
    )


def load_class_names(path: Path) -> list[str]:
    if not path.exists():
        raise RuntimeError(
            f"Class names file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        class_names = json.load(file)

    if not isinstance(class_names, list):
        raise RuntimeError(
            "Class names file must contain a JSON array."
        )

    if not all(isinstance(name, str) for name in class_names):
        raise RuntimeError(
            "Every class name must be a string."
        )

    if len(class_names) != NUM_CLASSES:
        raise RuntimeError(
            f"Expected {NUM_CLASSES} class names, "
            f"found {len(class_names)}."
        )

    if class_names != EXPECTED_CLASS_NAMES:
        raise RuntimeError(
            "class_names.json does not match the canonical 38-class ordering."
        )

    return class_names


def build_efficientnet_b0(
    num_classes: int = NUM_CLASSES,
) -> nn.Module:

    # Same architecture used in AgriVision.ipynb.
    #
    # During training the notebook used:
    #
    # model = models.efficientnet_b0(pretrained=True)
    #
    # and then replaced the classifier.
    #
    # At inference we use weights=None because the trained
    # weights are loaded from best_efficientnet_b0.pth.

    model = models.efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(
            p=0.2,
            inplace=True,
        ),
        nn.Linear(
            in_features,
            num_classes,
        ),
    )

    if model.classifier[1].out_features != NUM_CLASSES:
        raise RuntimeError(
            f"Classifier output dimension is {model.classifier[1].out_features}, "
            f"expected {NUM_CLASSES}."
        )

    return model


def load_model_bundle(settings: Settings) -> ModelBundle:

    model_path = settings.resolved_model_path

    if not model_path.exists():
        raise RuntimeError(
            f"Model checkpoint not found: {model_path}"
        )

    # ---------------------------------------------------------
    # Load the EXACT class ordering used during training
    # ---------------------------------------------------------

    class_names = load_class_names(
        settings.resolved_class_names_path
    )

    if len(class_names) != NUM_CLASSES:
        raise RuntimeError(
            f"Model expects {NUM_CLASSES} classes, "
            f"but class_names.json contains {len(class_names)}."
        )

    # ---------------------------------------------------------
    # Select device
    # ---------------------------------------------------------

    device = select_device(settings.device)

    # ---------------------------------------------------------
    # Build the same EfficientNet-B0 architecture
    # ---------------------------------------------------------

    model = build_efficientnet_b0(
        num_classes=len(class_names)
    )

    # ---------------------------------------------------------
    # Load trained checkpoint
    # ---------------------------------------------------------

    checkpoint = torch.load(
        model_path,
        map_location=device,
    )

    # The notebook saved:
    #
    # torch.save(model.state_dict(), BEST_MODEL_PATH)
    #
    # Therefore checkpoint should directly be a state_dict.

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Invalid model checkpoint format."
        )

    # Support both:
    #
    # 1. Direct state_dict
    # 2. {"state_dict": ...} checkpoint
    # 3. {"model_state_dict": ...} checkpoint
    #
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict) or not all(
        isinstance(key, str) and torch.is_tensor(value)
        for key, value in state_dict.items()
    ):
        raise RuntimeError("Checkpoint does not contain a valid model state_dict.")

    # Remove possible DataParallel prefix
    cleaned_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]

        cleaned_state_dict[key] = value

    # ---------------------------------------------------------
    # Load weights
    # ---------------------------------------------------------

    missing_keys, unexpected_keys = model.load_state_dict(
        cleaned_state_dict,
        strict=True,
    )

    if missing_keys:
        raise RuntimeError(
            f"Missing model weights: {missing_keys}"
        )

    if unexpected_keys:
        raise RuntimeError(
            f"Unexpected model weights: {unexpected_keys}"
        )

    # ---------------------------------------------------------
    # Move model to device
    # ---------------------------------------------------------

    model.to(device)

    # ---------------------------------------------------------
    # Inference mode
    # ---------------------------------------------------------

    model.eval()

    # ---------------------------------------------------------
    # Startup information
    # ---------------------------------------------------------

    print("=" * 60)
    print("AGRIVISION MODEL LOADED")
    print("=" * 60)

    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 224, 224, device=device))
    if tuple(output.shape) != (1, NUM_CLASSES):
        raise RuntimeError(
            f"Dummy forward produced shape {tuple(output.shape)}, expected (1, {NUM_CLASSES})."
        )

    print(f"MODEL VERSION: {settings.model_version.upper()}")
    print(f"MODEL PATH: {model_path}")
    print("ARCHITECTURE: EfficientNet-B0")
    print(f"NUM CLASSES: {len(class_names)}")
    print(f"CLASS MAPPING PATH: {settings.resolved_class_names_path}")
    print(f"DEVICE: {device}")
    print("INPUT SIZE: 224x224 (Resize 256, CenterCrop 224)")
    print("NORMALIZATION: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]")
    print("CLASS INDEX MAPPING:")
    for index, class_name in enumerate(class_names):
        print(f"{index} -> {class_name}")

    if torch.cuda.is_available():
        print(
            f"GPU         : "
            f"{torch.cuda.get_device_name(0)}"
        )
    else:
        print("GPU         : N/A")

    print("=" * 60)

    return ModelBundle(
        model=model,
        device=device,
        class_names=class_names,
        model_version=settings.model_version.lower().strip(),
        model_path=model_path,
    )