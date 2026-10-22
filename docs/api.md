# API reference (FastAPI)
Start: `uvicorn app.main:app` (or `docker compose up`). OpenAPI docs: `http://localhost:8000/docs`.

| Method | Path | Body | Response |
|---|---|---|---|
| GET | `/health` | — | `{"status": "ok"}` |
| POST | `/predict` | `{"sepal_length", "sepal_width", "petal_length", "petal_width"}` (floats, 0 < x < 20 cm) | `{"species": str, "probabilities": {species: float}}` |

Invalid input returns HTTP 422. If `models/iris_model.joblib` is missing, the API trains it on first use.
