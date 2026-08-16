# AgriVision Backend

FastAPI inference service for the existing PlantVillage EfficientNet-B0 checkpoint.

## Source of Truth

- Notebook: `../ml/notebooks/Welcome_To_Colab.ipynb`
- Checkpoint: `../ml/models/best_efficientnet_b0.pth`
- Class mapping source: `../ml/models/class_names.json`, copied to `app/class_names.json`

The backend uses the notebook's validation/test preprocessing:

1. Convert image to RGB
2. Resize to 256
3. Center crop to 224
4. Convert to tensor
5. Normalize with ImageNet mean and std

## Setup

```bash
cd backend
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```bash
cd backend
source .venv/bin/activate
python -m uvicorn app.main:app --reload
```

## Endpoints

- `GET /health`
- `POST /predict` with multipart form field `image`
- `POST /assistant/answer` for predefined disease questions
- `POST /assistant/ask` for custom disease questions

## AI Disease Assistant

The assistant uses the CNN result as structured context and generates agricultural guidance with a configurable open-source Hugging Face model. It does not replace the EfficientNet-B0 diagnosis and does not run during `/predict`.

Default local model:

```bash
HF_MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
```

This is a compact instruction-following model that is small enough for a local MVP while still producing practical farm guidance grounded in the CNN output.

Predefined question types:

- `overview`
- `symptoms`
- `causes`
- `severity`
- `management`
- `prevention`
- `immediate_actions`

Example:

```bash
curl -X POST http://localhost:8000/assistant/answer \
  -H "Content-Type: application/json" \
  -d '{"disease":"Grape___Leaf_blight_(Isariopsis_Leaf_Spot)","confidence":0.9983,"question_type":"symptoms"}'
```

Custom question example:

```bash
curl -X POST http://localhost:8000/assistant/ask \
  -H "Content-Type: application/json" \
  -d '{"disease":"Grape___Leaf_blight_(Isariopsis_Leaf_Spot)","confidence":0.9983,"question":"How can I prevent this disease next season?"}'
```
