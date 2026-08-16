from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    disease: str
    confidence: float = Field(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    prediction: PredictionItem
    top_predictions: list[PredictionItem]
    is_confident: bool
    message: str


class AssistantPredictionItem(BaseModel):
    disease: str
    confidence: float = Field(ge=0.0, le=1.0)


class AssistantAnswerRequest(BaseModel):
    crop: str = Field(min_length=1)
    disease: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    question_type: str = Field(min_length=1)
    top_predictions: list[AssistantPredictionItem] | None = None


class AssistantAnswerResponse(BaseModel):
    crop: str
    disease: str
    question_type: str
    answer: str


class AssistantAskRequest(BaseModel):
    crop: str = Field(min_length=1)
    disease: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    question: str = Field(min_length=1, max_length=1000)
    top_predictions: list[AssistantPredictionItem] | None = None


class AssistantAskResponse(BaseModel):
    crop: str
    disease: str
    question: str
    answer: str


class DiseaseSummary(BaseModel):
    id: str
    name: str


class ExplorerPlantsResponse(BaseModel):
    plants: list[str]


class ExplorerPlantDiseasesResponse(BaseModel):
    plant: str
    diseases: list[DiseaseSummary]


class ExplorerAnalyzeRequest(BaseModel):
    plant: str = Field(min_length=1)
    disease: str = Field(min_length=1)
    topics: list[str] = Field(min_length=1)


class ExplorerAnalyzeResponse(BaseModel):
    plant: str
    disease: str
    information: dict[str, str]
