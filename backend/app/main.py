import logging
from contextlib import asynccontextmanager
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from app.assistant import AssistantService
from app.config import get_settings
from app.explorer import ExplorerService
from app.model import ModelBundle, load_class_names, load_model_bundle
from app.predictor import predict_image
from app.schemas import (
    AssistantAnswerRequest,
    AssistantAnswerResponse,
    AssistantAskRequest,
    AssistantAskResponse,
    ExplorerAnalyzeRequest,
    ExplorerAnalyzeResponse,
    ExplorerPlantDiseasesResponse,
    PredictionResponse,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

model_bundle: ModelBundle | None = None
assistant_service: AssistantService | None = None
explorer_service: ExplorerService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global assistant_service, model_bundle, explorer_service
    settings = get_settings()

    model_bundle = load_model_bundle(settings)

    try:
        assistant_service = AssistantService(settings)
        explorer_service = ExplorerService(assistant_service, load_class_names(settings.resolved_class_names_path))
        logger.info("Assistant and explorer services loaded successfully.")
    except Exception as exc:
        logger.warning("Assistant model failed to load; continuing with CNN-only endpoints: %s", exc)
        assistant_service = None
        explorer_service = None

    yield


settings = get_settings()
app = FastAPI(title="AgriVision AI Backend", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(image: UploadFile = File(...)) -> dict:
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        contents = await image.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not read uploaded file.") from exc

    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(contents) > settings.max_upload_size_bytes:
        raise HTTPException(status_code=413, detail="Uploaded file exceeds the maximum allowed size.")

    try:
        image_file = Image.open(BytesIO(contents))
        image_file.verify()
        image_file = Image.open(BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid or corrupted image.")

    try:
        return predict_image(image_file, model_bundle, settings.confidence_threshold)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Inference failed.") from exc


@app.post("/assistant/answer", response_model=AssistantAnswerResponse)
def assistant_answer(request: AssistantAnswerRequest) -> dict[str, str]:
    if assistant_service is None:
        raise HTTPException(status_code=503, detail="Assistant model is not loaded.")

    try:
        answer = assistant_service.answer_predefined(
            crop=request.crop,
            disease=request.disease,
            confidence=request.confidence,
            question_type=request.question_type,
            top_predictions=request.top_predictions,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Assistant generation failed.") from exc

    return {
        "crop": request.crop,
        "disease": request.disease,
        "question_type": request.question_type,
        "answer": answer,
    }


@app.post("/assistant/ask", response_model=AssistantAskResponse)
def assistant_ask(request: AssistantAskRequest) -> dict[str, str]:
    if assistant_service is None:
        raise HTTPException(status_code=503, detail="Assistant model is not loaded.")

    try:
        answer = assistant_service.answer_custom(
            crop=request.crop,
            disease=request.disease,
            confidence=request.confidence,
            question=request.question,
            top_predictions=request.top_predictions,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Assistant generation failed.") from exc

    return {
        "crop": request.crop,
        "disease": request.disease,
        "question": request.question,
        "answer": answer,
    }


@app.get("/explorer/plants", response_model=list[str])
def explorer_plants() -> list[str]:
    if explorer_service is None:
        raise HTTPException(status_code=503, detail="Explorer service is not loaded.")
    return explorer_service.list_plants()


@app.get("/explorer/plants/{plant}/diseases", response_model=ExplorerPlantDiseasesResponse)
def explorer_plant_diseases(plant: str) -> dict[str, object]:
    if explorer_service is None:
        raise HTTPException(status_code=503, detail="Explorer service is not loaded.")
    try:
        return explorer_service.list_diseases_for_plant(plant)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/explorer/analyze", response_model=ExplorerAnalyzeResponse)
def explorer_analyze(request: ExplorerAnalyzeRequest) -> dict[str, object]:
    if explorer_service is None:
        raise HTTPException(status_code=503, detail="Explorer service is not loaded.")
    try:
        return explorer_service.analyze(request.plant, request.disease, request.topics)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Explorer analysis failed.") from exc
