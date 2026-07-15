from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_pipeline.joblib"

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Twitter Sentiment API",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1)


@app.get("/")
def root():
    return {"message": "Twitter Sentiment API is running"}


@app.post("/predict")
def predict(request: PredictionRequest):
    prediction = model.predict([request.text])[0]
    return {
        "text": request.text,
        "predicted_sentiment": str(prediction),
    }
