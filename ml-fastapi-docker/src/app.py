"""FastAPI 推理服务。"""

from functools import lru_cache

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .data_pipeline import FEATURES, TARGET_NAMES, model_path

app = FastAPI(title="Iris Classifier API", version="1.0.0")


class IrisRequest(BaseModel):
    sepal_length: float = Field(..., gt=0, description="萼片长度")
    sepal_width: float = Field(..., gt=0, description="萼片宽度")
    petal_length: float = Field(..., gt=0, description="花瓣长度")
    petal_width: float = Field(..., gt=0, description="花瓣宽度")


class PredictionResponse(BaseModel):
    predicted_class: str
    predicted_class_id: int
    probabilities: dict[str, float]


@lru_cache
def get_model():
    if not model_path().exists():
        raise RuntimeError("model.joblib 不存在，请先运行 python -m src.train")
    return joblib.load(model_path())


@app.get("/health")
def health() -> dict[str, str]:
    try:
        get_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict:
    return {"model": "LogisticRegression", "features": FEATURES, "classes": TARGET_NAMES}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: IrisRequest) -> PredictionResponse:
    try:
        model = get_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    row = pd.DataFrame([[getattr(request, feature) for feature in FEATURES]], columns=FEATURES)
    class_id = int(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0]
    return PredictionResponse(
        predicted_class=TARGET_NAMES[class_id],
        predicted_class_id=class_id,
        probabilities={
            TARGET_NAMES[i]: round(float(probability), 6)
            for i, probability in enumerate(probabilities)
        },
    )

