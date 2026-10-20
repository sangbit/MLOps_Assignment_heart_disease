"""REST API serving the heart-disease model (FastAPI).

Run locally (from the project root):
    uvicorn src.api:app --port 8000
Interactive docs: http://127.0.0.1:8000/docs

Endpoints
    GET  /          short description of the service
    GET  /health    is the service up and the model loaded? (used by Docker / Kubernetes)
    GET  /model     model metadata: type, version, training metrics
    POST /predict   one patient's raw values  ->  prediction + confidence
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src import predict as predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once when the server starts, so the first request isn't slow
    # and a missing/broken model stops the container immediately instead of on first use.
    predictor.load_model()
    predictor.load_model_card()
    yield


app = FastAPI(
    title="Heart Disease Risk API",
    description="Predicts the risk of heart disease from 13 clinical measurements. "
                "Educational project — not a medical device.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------- Request / response schemas (input validation happens here) ----------

class PatientInput(BaseModel):
    """Raw patient values. Only age and sex are required; anything unknown can be null or omitted."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 2,
        "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6}})

    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int | None = Field(None, ge=1, le=4, description="Chest pain type: 1 typical angina, "
                           "2 atypical angina, 3 non-anginal, 4 asymptomatic")
    trestbps: float | None = Field(None, ge=0, le=300, description="Resting blood pressure, mm Hg")
    chol: float | None = Field(None, ge=0, le=700, description="Serum cholesterol, mg/dl")
    fbs: int | None = Field(None, ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1 = yes)")
    restecg: int | None = Field(None, ge=0, le=2, description="Resting ECG: 0 normal, "
                                "1 ST-T abnormality, 2 LV hypertrophy")
    thalach: float | None = Field(None, ge=40, le=250, description="Maximum heart rate achieved")
    exang: int | None = Field(None, ge=0, le=1, description="Exercise-induced angina (1 = yes)")
    oldpeak: float | None = Field(None, ge=-5, le=10, description="ST depression, exercise vs rest")
    slope: int | None = Field(None, ge=1, le=3, description="ST slope: 1 up, 2 flat, 3 down")
    ca: int | None = Field(None, ge=0, le=3, description="Major vessels coloured by fluoroscopy (0-3)")
    thal: int | None = Field(None, description="Thallium test: 3 normal, 6 fixed, 7 reversible defect")

    @field_validator("thal")
    @classmethod
    def thal_allowed_values(cls, v):
        if v is not None and v not in (3, 6, 7):
            raise ValueError("thal must be 3, 6 or 7")
        return v


class PredictionOutput(BaseModel):
    prediction: int = Field(..., description="1 = heart disease, 0 = no heart disease")
    label: str = Field(..., description="'Disease' or 'No disease'")
    confidence: float = Field(..., description="Probability of the predicted class (0.5-1.0)")
    disease_probability: float = Field(..., description="Probability of heart disease (0-1)")
    model_version: str = Field(..., description="When the served model was trained")


# ---------- Endpoints ----------

@app.get("/")
def root():
    return {"service": "Heart Disease Risk API", "docs": "/docs", "predict": "POST /predict"}


@app.get("/health")
def health():
    try:
        predictor.load_model()
    except Exception as exc:     # model missing or unreadable
        raise HTTPException(status_code=503, detail=f"Model not available: {exc}") from exc
    return {"status": "ok", "model_loaded": True}


@app.get("/model")
def model_info():
    card = predictor.load_model_card()
    return {key: card.get(key) for key in
            ("model_name", "model_type", "model_params", "created_at", "decision_threshold",
             "metrics", "lineage")}


@app.post("/predict", response_model=PredictionOutput)
def predict(patient: PatientInput):
    result = predictor.predict(patient.model_dump())[0]
    return {**result, "model_version": predictor.load_model_card().get("created_at", "unknown")}
