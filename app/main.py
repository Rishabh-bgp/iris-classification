"""FastAPI service exposing the trained Iris classifier."""
from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.data import SPECIES

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "iris_model.joblib"
app = FastAPI(title="Iris Classification API", version="1.0.0")
_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            from src.train import train
            train()
        _model = joblib.load(MODEL_PATH)
    return _model


class Flower(BaseModel):
    sepal_length: float = Field(..., gt=0, lt=20, examples=[5.1])
    sepal_width: float = Field(..., gt=0, lt=20, examples=[3.5])
    petal_length: float = Field(..., gt=0, lt=20, examples=[1.4])
    petal_width: float = Field(..., gt=0, lt=20, examples=[0.2])


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(f: Flower) -> dict:
    x = [[f.sepal_length, f.sepal_width, f.petal_length, f.petal_width]]
    proba = get_model().predict_proba(x)[0]
    best = int(proba.argmax())
    return {"species": SPECIES[best],
            "probabilities": {s: round(float(p), 6) for s, p in zip(SPECIES, proba)}}
