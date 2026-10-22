import json
import math
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from src.data import load_iris_df

client = TestClient(app)
ROOT = Path(__file__).resolve().parent.parent


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_setosa():
    r = client.post("/predict", json={"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2})
    assert r.status_code == 200
    body = r.json()
    assert body["species"] == "Iris-setosa"
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-4


def test_predict_virginica():
    r = client.post("/predict", json={"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.8, "petal_width": 2.2})
    assert r.json()["species"] == "Iris-virginica"


def test_validation_error():
    assert client.post("/predict", json={"sepal_length": -1}).status_code == 422


def test_dataset_shape():
    df = load_iris_df()
    assert df.shape == (150, 5)
    assert df["species"].value_counts().tolist() == [50, 50, 50]


def _js_style_predict(p, x):
    z = [(xi - m) / s for xi, m, s in zip(x, p["mean"], p["scale"])]
    logits = [sum(w * zi for w, zi in zip(row, z)) + b for row, b in zip(p["coef"], p["intercept"])]
    mx = max(logits)
    e = [math.exp(v - mx) for v in logits]
    return [v / sum(e) for v in e]


def test_exported_json_matches_sklearn():
    client.get("/health")
    from app.main import get_model
    model = get_model()
    p = json.loads((ROOT / "web" / "model.json").read_text())
    df = load_iris_df()
    for x in df.iloc[:, :4].to_numpy():
        ref = model.predict_proba([x])[0]
        mine = _js_style_predict(p, x)
        assert max(abs(a - b) for a, b in zip(ref, mine)) < 1e-9
