from contextlib import asynccontextmanager
from pathlib import Path
import sys

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PHASE_1 = REPOSITORY_ROOT / "phase-1-machine-learning-nlp"
TASK_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = TASK_ROOT / "models" / "sentiment_pipeline.joblib"

sys.path.insert(0, str(PHASE_1 / "02-preprocessing-tokenization" / "src"))
sys.path.insert(0, str(PHASE_1 / "06-vectorization" / "src"))

model_store = {}


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, value):
        if not value.strip():
            raise ValueError("Text must contain a non-whitespace character.")
        return value


class PredictionResponse(BaseModel):
    text: str
    predicted_sentiment: str
    decision_score: float


@asynccontextmanager
async def lifespan(app):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Train the model before starting the API: {MODEL_PATH}")
    model_store["pipeline"] = joblib.load(MODEL_PATH)
    yield
    model_store.clear()


app = FastAPI(
    title="IMDB Sentiment API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "pipeline" in model_store}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    pipeline = model_store["pipeline"]
    features = pipeline.named_steps["tfidf"].transform([request.text])
    classifier = pipeline.named_steps["classifier"]
    prediction = int(classifier.predict(features)[0])
    score = float(classifier.decision_function(features)[0])
    return {
        "text": request.text,
        "predicted_sentiment": "positive" if prediction == 1 else "negative",
        "decision_score": round(score, 6),
    }
