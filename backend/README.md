# AgriVision Backend

A **FastAPI-based AI inference service** for plant disease detection and agricultural guidance. Combines EfficientNet-B0 CNN for disease classification with LLM-powered agricultural insights.

## Project Overview

AgriVision is an end-to-end agricultural disease detection system that:
- **Detects plant diseases** from images using a trained EfficientNet-B0 model on the PlantVillage dataset
- **Provides AI-driven guidance** on disease management, prevention, and immediate actions using a lightweight Qwen LLM
- **Explores plant diseases** through a structured catalog with semantic search and analysis
- **Supports multiple endpoints** for prediction, assistance, and disease exploration

## Architecture

```
AgriVision/
├── backend/          # FastAPI service (this folder)
│   ├── app/
│   │   ├── main.py              # FastAPI application & routes
│   │   ├── model.py             # EfficientNet-B0 model loading & inference
│   │   ├── predictor.py         # Image preprocessing & prediction logic
│   │   ├── assistant.py         # LLM-powered disease advisor
│   │   ├── explorer.py          # Plant disease catalog & semantic search
│   │   ├── config.py            # Settings management
│   │   ├── schemas.py           # Pydantic request/response models
│   │   ├── knowledge.py         # Disease knowledge base
│   │   ├── class_names.json     # Plant-disease class mapping
│   │   └── *_prompts.py         # LLM prompt templates
│   └── requirements.txt         # Python dependencies
├── ml/               # Machine learning models & notebooks
│   ├── models/
│   │   └── best_efficientnet_b0.pth  # Trained model checkpoint
│   └── notebooks/
│       └── Welcome_To_Colab.ipynb    # Training & validation notebook
```

## Core Features

### 1. **Disease Prediction** (`/predict`)
- Accepts leaf/plant images via HTTP POST
- Outputs disease classification with confidence scores
- Uses ImageNet-normalized preprocessing (256→224 center crop)

### 2. **AI Disease Assistant** (`/assistant/answer`, `/assistant/ask`)
- Generates agricultural guidance on diseases
- Supports predefined question types: `overview`, `symptoms`, `causes`, `severity`, `management`, `prevention`, `immediate_actions`
- Uses Qwen 2.5 0.5B (configurable) with context from CNN output
- Lightweight enough to run locally on CPU/GPU

### 3. **Plant Disease Explorer** (`/explorer/*`)
- Browse all supported plants and diseases
- Semantic search with LLM-generated insights
- Structured knowledge base with disease details

## Setup & Installation

### Prerequisites
- **Python 3.12+**
- **PyTorch 2.13+** (included in requirements)
- Optional: **CUDA 13.0** for GPU acceleration
- Optional: **HuggingFace token** for model access

### Installation

```bash
cd backend
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the `backend/` directory:

```env
# Model settings
MODEL_VERSION=v2
MODEL_PATH_V1=../ml/models/best_efficientnet_b0.pth
MODEL_PATH_V2=../ml/models/best_efficientnet_b0_v2.pth
CLASS_NAMES_PATH=app/class_names.json
DEVICE=auto                    # auto, cpu, cuda, mps
CONFIDENCE_THRESHOLD=0.60

# Upload limits
MAX_UPLOAD_SIZE_BYTES=10485760 # 10MB

# CORS origins
CORS_ORIGINS=http://localhost:3000,http://localhost:5173

# LLM settings
HF_MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
HF_TOKEN=                      # Optional: your HuggingFace token
ASSISTANT_DEVICE=auto
ASSISTANT_MAX_NEW_TOKENS=320
ASSISTANT_TEMPERATURE=0.2
ASSISTANT_LOW_CONFIDENCE_THRESHOLD=0.60
```

## Running the Service

```bash
cd backend
source .venv/bin/activate
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000/docs` for interactive API documentation (Swagger UI).

## API Endpoints

### Health Check
```bash
GET /health
```

### Disease Prediction
```bash
POST /predict
Content-Type: multipart/form-data

# Upload an image file with field name "image"
curl -X POST http://localhost:8000/predict \
  -F "image=@path/to/leaf.jpg"

# Response:
{
  "disease": "Tomato___Late_blight",
  "confidence": 0.9856,
  "class_name": "Tomato___Late_blight"
}
```

### Assistant - Predefined Questions
```bash
POST /assistant/answer
Content-Type: application/json

curl -X POST http://localhost:8000/assistant/answer \
  -H "Content-Type: application/json" \
  -d '{
    "disease": "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "confidence": 0.9983,
    "question_type": "management"
  }'

# Response:
{
  "disease": "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
  "confidence": 0.9983,
  "question_type": "management",
  "answer": "For managing leaf blight on grapes, ..."
}
```

**Supported question types:**
- `overview` - General disease information
- `symptoms` - Visual & biological symptoms
- `causes` - Environmental & biological causes
- `severity` - Potential crop impact
- `management` - Treatment & control strategies
- `prevention` - Preventive measures
- `immediate_actions` - Emergency response steps

### Assistant - Custom Questions
```bash
POST /assistant/ask
Content-Type: application/json

curl -X POST http://localhost:8000/assistant/ask \
  -H "Content-Type: application/json" \
  -d '{
    "disease": "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "confidence": 0.9983,
    "question": "How can I prevent this disease next season?"
  }'

# Response:
{
  "disease": "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
  "confidence": 0.9983,
  "question": "How can I prevent this disease next season?",
  "answer": "To prevent leaf blight next season, ..."
}
```

### Explorer - List Plants
```bash
GET /explorer/plants

# Response:
["Apple", "Blueberry", "Cherry", "Corn", ...]
```

### Explorer - List Diseases for a Plant
```bash
GET /explorer/diseases?plant=Tomato

# Response:
{
  "plant": "Tomato",
  "diseases": [
    {"id": "tomato_bacterial_spot", "name": "Bacterial Spot"},
    {"id": "tomato_early_blight", "name": "Early Blight"},
    ...
  ]
}
```

### Explorer - Analyze Plant Disease
```bash
POST /explorer/analyze
Content-Type: application/json

curl -X POST http://localhost:8000/explorer/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "plant": "Tomato",
    "disease": "Late Blight",
    "topics": ["symptoms", "management", "prevention"]
  }'

# Response:
{
  "plant": "Tomato",
  "disease": "Late Blight",
  "analysis": {
    "symptoms": "...",
    "management": "...",
    "prevention": "..."
  }
}
```

## Model Details

### CNN Model: EfficientNet-B0
- **Training Dataset:** PlantVillage (38 plant-disease classes)
- **Input:** RGB images, 224×224 pixels
- **Preprocessing:** ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Checkpoint:** `../ml/models/best_efficientnet_b0.pth`
- **Class Mapping:** `../ml/models/class_names.json`

### LLM: Qwen 2.5 0.5B-Instruct (Default)
- **Model:** `Qwen/Qwen2.5-0.5B-Instruct` from HuggingFace Hub
- **Size:** ~600M parameters (lightweight, CPU-friendly)
- **Temperature:** 0.2 (deterministic output)
- **Max Tokens:** 320 (concise responses)
- **Context:** Disease name + CNN confidence as structured input

**Alternative models** can be configured via `HF_MODEL_ID` environment variable.

## Preprocessing Pipeline

Images are processed identically to validation/test data from the training notebook:
1. Convert to RGB
2. Resize to 256×256
3. Center crop to 224×224
4. Convert to tensor
5. Normalize with ImageNet statistics

## Performance & Deployment

- **Latency:** ~100-500ms per prediction (depends on device & image size)
- **Memory:** ~2GB (model + LLM on GPU), ~4GB+ (CPU)
- **Concurrency:** Async FastAPI handles multiple requests efficiently
- **CORS:** Configured for localhost (dev) and configurable origins

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Model fails to load | Ensure `MODEL_PATH` points to a valid `.pth` file; check file exists and CUDA/CPU device is available |
| LLM fails to load | HuggingFace token may be required for gated models; set `HF_TOKEN` env var |
| Out of memory | Reduce `ASSISTANT_MAX_NEW_TOKENS` or use CPU if GPU is OOM |
| CORS errors | Add frontend origin to `CORS_ORIGINS` in `.env` |
| Slow inference | Ensure CUDA is installed and `DEVICE` is set to `cuda` (not `cpu`) |

## Development

### Source of Truth
- **Model:** `../ml/models/best_efficientnet_b0.pth`
- **Classes:** `../ml/models/class_names.json`
- **Training Notebook:** `../ml/notebooks/Welcome_To_Colab.ipynb`

### Running Tests
```bash
pytest tests/  # (Add tests as needed)
```

### Adding New Models
1. Update `MODEL_PATH` in config
2. Ensure new model uses same preprocessing pipeline
3. Update class mapping if classes change

## License

[Add license information if applicable]

## Contributing

[Add contribution guidelines if applicable]
