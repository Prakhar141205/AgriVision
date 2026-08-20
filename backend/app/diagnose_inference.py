import argparse
from pathlib import Path

from PIL import Image

from app.config import Settings
from app.model import load_model_bundle
from app.predictor import predict_image


def run_diagnostic(image_path: Path, model_version: str) -> None:
    with Image.open(image_path) as image:
        original_size = image.size
        original_mode = image.mode
        rgb_image = image.convert("RGB")

    settings = Settings(model_version=model_version)
    bundle = load_model_bundle(settings)
    result = predict_image(rgb_image, bundle, settings.confidence_threshold)

    print(f"MODEL VERSION: {bundle.model_version.upper()}")
    print(f"MODEL PATH: {bundle.model_path}")
    print(f"IMAGE ORIGINAL SIZE: {original_size}")
    print(f"IMAGE MODE: {original_mode}")
    print("PREPROCESSED TENSOR SHAPE: (1, 3, 224, 224)")
    print("TOP 5 PREDICTIONS:")
    for item in result["top_predictions"]:
        print(f"  {item['disease']}: {item['confidence']:.6f}")
    print(f"TOP 1 CONFIDENCE: {result['prediction']['confidence']:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AgriVision local inference diagnostics.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model-version", choices=("v1", "v2"), default="v2")
    args = parser.parse_args()
    run_diagnostic(args.image, args.model_version)


if __name__ == "__main__":
    main()