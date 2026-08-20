from PIL import Image
import torch
from torchvision import transforms

from app.model import ModelBundle


# ============================================================
# AGRIVISION - INFERENCE PREPROCESSING
# Must exactly match agrivision.ipynb validation/test pipeline
# ============================================================

INPUT_SIZE = 224

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


INFERENCE_TRANSFORM = transforms.Compose(
    [
        # EXACTLY same as val_test_transforms_corrected
        transforms.Resize(256),
        transforms.CenterCrop(INPUT_SIZE),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=NORM_MEAN,
            std=NORM_STD
        ),
    ]
)


def predict_image(
    image: Image.Image,
    bundle: ModelBundle,
    confidence_threshold: float = 0.60,
) -> dict:

    # --------------------------------------------------------
    # 1. Apply EXACT notebook preprocessing
    # --------------------------------------------------------

    image = image.convert("RGB")

    tensor = INFERENCE_TRANSFORM(image)

    # Add batch dimension
    tensor = tensor.unsqueeze(0)

    # Move to same device as model
    tensor = tensor.to(bundle.device)

    # --------------------------------------------------------
    # 2. Model inference
    # --------------------------------------------------------

    bundle.model.eval()

    with torch.no_grad():

        logits = bundle.model(tensor)

        probabilities = torch.softmax(
            logits,
            dim=1
        ).squeeze(0)

    # --------------------------------------------------------
    # 3. Top-5 predictions
    # --------------------------------------------------------

    top_k = min(5, len(bundle.class_names))

    confidences, indices = torch.topk(
        probabilities,
        k=top_k
    )

    top_predictions = []

    for confidence, index in zip(
        confidences.cpu(),
        indices.cpu()
    ):

        class_index = index.item()

        top_predictions.append(
            {
                "disease": bundle.class_names[class_index],
                "confidence": float(confidence.item()),
            }
        )

    # --------------------------------------------------------
    # 4. Primary prediction
    # --------------------------------------------------------

    top_prediction = top_predictions[0]

    is_confident = (
        top_prediction["confidence"]
        >= confidence_threshold
    )

    # --------------------------------------------------------
    # 5. Confidence message
    # --------------------------------------------------------

    if is_confident:

        message = (
            "The model identified the disease "
            "with a confident prediction."
        )

    else:

        message = (
            "The model could not confidently identify "
            "the disease. Consider uploading a clearer "
            "image or consulting an agricultural expert."
        )

    # --------------------------------------------------------
    # 6. API response
    # --------------------------------------------------------

    return {
        "prediction": top_prediction,
        "top_predictions": top_predictions,
        "is_confident": is_confident,
        "message": message,
    }