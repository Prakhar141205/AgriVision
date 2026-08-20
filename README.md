# AgriVision

🌾 **AI-Powered Plant Disease Detection & Agricultural Guidance System**

AgriVision is an intelligent agricultural technology platform that combines computer vision and large language models to detect plant diseases from images and provide actionable farming recommendations.

## 🎯 Features

- ✅ **Accurate Disease Detection** - EfficientNet-B0 CNN trained on 38 plant-disease classes from PlantVillage dataset
- ✅ **AI-Driven Guidance** - Lightweight Qwen LLM provides context-aware agricultural insights
- ✅ **Disease Knowledge Explorer** - Browse supported plants, diseases, and management strategies
- ✅ **Production-Ready API** - FastAPI backend with async processing, CORS support, and error handling
- ✅ **Local Model Support** - Run inference locally without cloud dependencies
- ✅ **Flexible Configuration** - Environment-based settings for easy deployment across environments

## 📁 Project Structure

```
AgriVision/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application & route definitions
│   │   ├── model.py             # EfficientNet-B0 model initialization & loading
│   │   ├── predictor.py         # Image preprocessing & inference pipeline
│   │   ├── assistant.py         # LLM-powered disease advisor
│   │   ├── explorer.py          # Plant disease catalog & search
│   │   ├── config.py            # Environment settings & configuration
│   │   ├── schemas.py           # Pydantic request/response models
│   │   ├── knowledge.py         # Disease knowledge base
│   │   ├── class_names.json     # Plant-disease class mappings (38 classes)
│   │   ├── assistant_prompts.py # Disease guidance prompt templates
│   │   ├── explorer_prompts.py  # Disease analysis prompt templates
│   │   └── __init__.py
│   ├── requirements.txt         # Python dependencies (FastAPI, PyTorch, Transformers, etc.)
│   └── README.md               # Backend documentation
│
├── ml/
│   ├── models/
│   │   ├── best_efficientnet_b0.pth  # Trained model checkpoint (~90MB)
│   │   └── class_names.json          # Class label definitions
│   ├── notebooks/
│   │   └── Welcome_To_Colab.ipynb   # Model training & validation notebook
│   ├── data/                         # Training/validation datasets (if included)
│   └── src/                          # Training utilities & helper functions
│
├── README.md                    # This file
└── .gitignore
```

## 🚀 Quick Start

### 1. Backend Setup (5 minutes)

```bash
cd backend
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

Create `backend/.env`:
```env
MODEL_VERSION=v2
MODEL_PATH_V1=../ml/models/best_efficientnet_b0.pth
MODEL_PATH_V2=../ml/models/best_efficientnet_b0_v2.pth
CLASS_NAMES_PATH=app/class_names.json
DEVICE=auto
HF_MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
```

### 3. Run the Server

```bash
python -m uvicorn app.main:app --reload
```

Access the API at:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **API Base:** http://localhost:8000

## 📡 API Overview

### Core Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Service health check |
| `/predict` | POST | Classify disease from image |
| `/assistant/answer` | POST | Get predefined disease answers |
| `/assistant/ask` | POST | Ask custom disease questions |
| `/explorer/plants` | GET | List all supported plants |
| `/explorer/diseases` | GET | Get diseases for a plant |
| `/explorer/analyze` | POST | Analyze disease with LLM |

### Example: Predict Disease

```bash
curl -X POST http://localhost:8000/predict \
  -F "image=@path/to/leaf.jpg"

# Response:
{
  "disease": "Tomato___Late_blight",
  "confidence": 0.9856,
  "class_name": "Tomato___Late_blight"
}
```

### Example: Get Disease Guidance

```bash
curl -X POST http://localhost:8000/assistant/answer \
  -H "Content-Type: application/json" \
  -d '{
    "disease": "Tomato___Late_blight",
    "confidence": 0.9856,
    "question_type": "management"
  }'

# Response:
{
  "disease": "Tomato___Late_blight",
  "confidence": 0.9856,
  "question_type": "management",
  "answer": "To manage late blight on tomatoes, ..."
}
```

### Example: Explore Diseases

```bash
# Get all plants
curl http://localhost:8000/explorer/plants

# Get diseases for a plant
curl "http://localhost:8000/explorer/diseases?plant=Tomato"

# Analyze a disease
curl -X POST http://localhost:8000/explorer/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "plant": "Tomato",
    "disease": "Late Blight",
    "topics": ["symptoms", "management", "prevention"]
  }'
```

See [backend/README.md](backend/README.md) for complete API documentation.

## 🤖 AI Models

### Computer Vision: EfficientNet-B0
- **Architecture:** Efficient CNN for mobile/edge deployment
- **Training Data:** PlantVillage dataset (38 plant-disease classes)
- **Input:** 224×224 RGB images
- **Performance:** High accuracy with fast inference
- **Checkpoint:** `ml/models/best_efficientnet_b0.pth`

### Language Model: Qwen 2.5 0.5B-Instruct
- **Size:** ~600M parameters (runs on CPU)
- **Purpose:** Generate agricultural guidance from disease predictions
- **Context:** Disease name + confidence score → actionable farm advice
- **Customizable:** Set `HF_MODEL_ID` to use different models

## ⚙️ Configuration

All settings are managed via environment variables in `backend/.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_VERSION` | `v2` | Explicit checkpoint selection (`v1` or `v2`) |
| `MODEL_PATH_V1` | `../ml/models/best_efficientnet_b0.pth` | V1 EfficientNet checkpoint |
| `MODEL_PATH_V2` | `../ml/models/best_efficientnet_b0_v2.pth` | V2 EfficientNet checkpoint |
| `CLASS_NAMES_PATH` | `app/class_names.json` | Path to class label mappings |
| `DEVICE` | `auto` | Compute device (auto/cpu/cuda/mps) |
| `CONFIDENCE_THRESHOLD` | `0.60` | Minimum confidence for predictions |
| `MAX_UPLOAD_SIZE_BYTES` | `10485760` | Max image size (10MB default) |
| `HF_MODEL_ID` | `Qwen/Qwen2.5-0.5B-Instruct` | LLM model from HuggingFace |
| `HF_TOKEN` | (empty) | HuggingFace API token (optional) |
| `CORS_ORIGINS` | `http://localhost:3000,...` | Allowed frontend origins |
| `ASSISTANT_DEVICE` | `auto` | LLM compute device |
| `ASSISTANT_MAX_NEW_TOKENS` | `320` | Max LLM output length |
| `ASSISTANT_TEMPERATURE` | `0.2` | LLM output randomness |

## 📊 Supported Plants & Diseases

AgriVision supports **38 plant-disease combinations** including:

**Crops:** Tomato, Potato, Grape, Corn, Cotton, Rice, Sugarcane, Wheat
**Fruits:** Apple, Blueberry, Cherry, Peach, Raspberry, Strawberry, Orange
**Vegetables:** Bell Pepper, Cucumber, Squash

Access the full catalog via `/explorer/plants` endpoint.

## 🔧 Development

### Running Tests

```bash
cd backend
pytest tests/  # (add tests as needed)
```

### Adding New Models

1. Replace checkpoint in `ml/models/best_efficientnet_b0.pth`
2. Update class labels in `ml/models/class_names.json`
3. Ensure preprocessing pipeline matches (224×224, ImageNet norm)
4. Update `MODEL_PATH` in `.env`

### Debugging

```bash
# Enable verbose logging
export LOGLEVEL=DEBUG
python -m uvicorn app.main:app --log-level debug

# Check model loading
python -c "from app.model import load_model_bundle; load_model_bundle()"

# Verify LLM
python -c "from app.assistant import AssistantService; AssistantService()"
```

## 📦 Dependencies

### Core
- **FastAPI** - Web framework
- **Uvicorn** - ASGI server
- **PyTorch** - Deep learning (EfficientNet)
- **TorchVision** - Image processing & preprocessing
- **Transformers** - LLM inference (Qwen)
- **Pillow** - Image manipulation
- **Pydantic** - Data validation & settings

See `backend/requirements.txt` for full list.

## 🚀 Deployment

### Docker (Recommended)

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/app app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t agrivision-backend .
docker run -p 8000:8000 -v /path/to/models:/app/models agrivision-backend
```

### Local Development

```bash
# With auto-reload
python -m uvicorn app.main:app --reload

# Production
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Environment Considerations

- **GPU:** Set `DEVICE=cuda` for acceleration (requires CUDA 13.0+)
- **CPU:** Set `DEVICE=cpu` for compatibility
- **Memory:** ~2GB (GPU), ~4GB+ (CPU)

## 📖 Documentation

- **Backend API:** See [backend/README.md](backend/README.md) for complete endpoint documentation
- **Model Training:** See `ml/notebooks/Welcome_To_Colab.ipynb` for training details
- **Configuration:** See `.env.example` (if present) for all available settings

## 🔒 Security Considerations

- Image upload size limits (configurable via `MAX_UPLOAD_SIZE_BYTES`)
- CORS restrictions (configure `CORS_ORIGINS` for production)
- No credential/token storage in code (use `.env` file)
- Model runs locally (no data sent to external services, unless using gated HF models)

## ⚡ Performance

- **Prediction Latency:** 100-500ms (device dependent)
- **Memory Footprint:** 2GB (GPU) / 4GB+ (CPU)
- **Concurrent Requests:** Limited by available VRAM; use async workers for scaling
- **Throughput:** ~2-10 predictions/sec (single GPU)

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| `Model not found` | Verify `MODEL_PATH` exists and file is valid `.pth` |
| `CUDA out of memory` | Reduce `ASSISTANT_MAX_NEW_TOKENS` or use CPU |
| `LLM fails to load` | Check HuggingFace token (`HF_TOKEN`) for gated models |
| `CORS errors` | Add your frontend URL to `CORS_ORIGINS` |
| `Slow inference` | Ensure CUDA is properly installed (`DEVICE=cuda`) |

## 📝 License

[Add license information]

## 🤝 Contributing

[Add contribution guidelines]

## 📧 Contact & Support

[Add contact information]

---

**Made with 🌾 for sustainable agriculture**
