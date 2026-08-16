from PIL import Image
import torch
from torchvision import transforms

from app.model import ModelBundle


INFERENCE_TRANSFORM = transforms.Compose(
    [
        transforms.Lambda(lambda image: image.convert("RGB")),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def predict_image(image: Image.Image, bundle: ModelBundle, confidence_threshold: float = 0.60) -> dict:
    tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(bundle.device)

    with torch.no_grad():
        logits = bundle.model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        confidences, indices = torch.topk(probabilities, k=5)

    top_predictions = [
        {
            "disease": bundle.class_names[index.item()],
            "confidence": float(confidence.item()),
        }
        for confidence, index in zip(confidences.cpu(), indices.cpu())
    ]

    top_prediction = top_predictions[0]
    is_confident = top_prediction["confidence"] >= confidence_threshold
    message = (
        "The model could not confidently identify the disease. Consider uploading a clearer image or consulting an agricultural expert."
        if not is_confident
        else "The model identified the disease with a confident prediction."
    )

    return {
        "prediction": top_prediction,
        "top_predictions": top_predictions,
        "is_confident": is_confident,
        "message": message,
    }
