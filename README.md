<div align="center">

# 🌸 Iris Flower Classification — Machine Learning in Python (scikit-learn, FastAPI) with a Live In-Browser Demo

[![Live demo](https://img.shields.io/badge/live%20demo-GitHub%20Pages-e75480?logo=github)](https://rishabh-bgp.github.io/iris-classification/)
[![CI](https://github.com/Rishabh-bgp/iris-classification/actions/workflows/ci.yml/badge.svg)](https://github.com/Rishabh-bgp/iris-classification/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.1-F7931E?logo=scikitlearn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.160-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-b79ced)

**[▶ Try the live demo](https://rishabh-bgp.github.io/iris-classification/)**

<img src="docs/img/demo-virginica.png" alt="Screenshot of the pastel sakura-themed Iris classifier demo: four measurement sliders, a chibi mascot with a speech bubble predicting virginica, species cards with probabilities, and a petal scatter plot" width="900"/>

</div>

An end-to-end **machine learning project in Python**: exploratory data analysis of the classic **UCI Iris dataset**,
comparison of eight **scikit-learn** classifiers with stratified cross-validation, hyperparameter tuning, evaluation,
interpretation, a **FastAPI `/predict` REST API**, **Docker** deployment, **GitHub Actions CI** and an interactive web demo
on **GitHub Pages** where the trained model runs entirely in your browser (JavaScript inference from exported parameters).

> Research work carried out as part of the **6-Month Industrial Training & Certification Program in Artificial Intelligence and Machine Learning**, jointly offered by [IIT Ropar](https://www.iitrpr.ac.in/) and [NIELIT Ropar](https://www.nielit.gov.in/ropar/), under MeitY, Govt. of India.

---

## 📈 Results (from the executed notebook)

10-fold stratified cross-validation, default hyperparameters, standard scaling in each pipeline:

| Model | CV accuracy | SD |
|---|---|---|
| LDA | **0.973** | 0.044 |
| K-nearest neighbours | 0.960 | 0.053 |
| SVM (RBF) | 0.960 | 0.044 |
| Logistic regression | 0.953 | 0.052 |
| Random forest | 0.953 | 0.052 |
| Gaussian naive Bayes | 0.953 | 0.052 |
| Decision tree | 0.933 | 0.052 |
| Majority baseline | 0.333 | 0.000 |

**Tuning (GridSearchCV, same folds):** logistic regression 0.973 (C = 100) · KNN 0.973 (k = 13) · SVM 0.967 (C = 1, γ = 0.1).

**Deployed model — tuned logistic regression:**
* Out-of-fold accuracy **0.973** (150 flowers); macro F1 0.973. Setosa 1.000 F1; versicolor 0.959; virginica 0.961.
* Stratified 70/30 hold-out: accuracy 0.911 (45 flowers) — a small split, so noisier than CV.
* Most informative features: petal length and petal width (random-forest importances 0.455 and 0.417; largest
  logistic-regression coefficients).

**Browser ↔ Python check:** the JavaScript model on the live page reproduces scikit-learn's `predict_proba` on all 150
flowers with a maximum absolute difference of 4.4 × 10⁻¹⁶ and 100% label agreement.

Details: [docs/results.md](docs/results.md).

## 🖼️ Screenshots

| Desktop | Mobile |
|---|---|
| <img src="docs/img/demo-setosa.png" alt="Desktop view of the demo predicting setosa" width="520"/> | <img src="docs/img/demo-mobile.png" alt="Mobile view of the demo" width="220"/> |

## 🏗️ Architecture

```mermaid
flowchart LR
  A[data/iris.data<br/>UCI Iris, CC BY 4.0] --> B[src/train.py<br/>StandardScaler + LogisticRegression<br/>GridSearchCV]
  B --> C[models/iris_model.joblib]
  B --> D[web/model.json<br/>mean, scale, coef, intercept]
  C --> E[app/main.py<br/>FastAPI /predict]
  D --> F[web/ app.js<br/>softmax inference in the browser]
  F --> G[GitHub Pages live demo]
  E --> H[Docker / docker-compose]
```

## 📂 Repository structure
```
notebooks/iris_classification.ipynb   EDA, preprocessing, model comparison, tuning, evaluation, interpretation
src/data.py, src/train.py             data loading; training + export (joblib and JSON)
app/main.py                           FastAPI app: GET /health, POST /predict
tests/test_api.py                     pytest: API, validation, dataset, JSON-export parity with scikit-learn
web/                                  static demo (index.html, style.css, app.js, model.json, iris.json)
Dockerfile, docker-compose.yml        local deployment (API on :8000, demo on :8080)
.github/workflows/ci.yml              CI: ruff + pytest on Python 3.12 and 3.13
gh-pages branch                       published copy of web/ served by GitHub Pages
docs/                                 extended documentation
```

## 🚀 Run locally
```bash
git clone https://github.com/Rishabh-bgp/iris-classification.git
cd iris-classification
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m src.train                       # trains, writes models/ and web/model.json
pytest -q                                 # 6 tests
uvicorn app.main:app --reload             # API at http://localhost:8000/docs
python -m http.server -d web 8080         # demo at http://localhost:8080
```

## 🐳 Docker
```bash
docker compose up --build                 # API → http://localhost:8000 · demo → http://localhost:8080
```

## 🔌 API
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```
Returns `{"species": "Iris-setosa", "probabilities": {...}}`. Interactive docs at `/docs`. See [docs/api.md](docs/api.md).

## 📚 Documentation
| Page | Contents |
|---|---|
| [docs/data.md](docs/data.md) | Dataset, attribution, licence |
| [docs/methodology.md](docs/methodology.md) | Pipeline, models, validation, tuning |
| [docs/results.md](docs/results.md) | Full results and interpretation |
| [docs/api.md](docs/api.md) | REST API reference |
| [docs/web-demo.md](docs/web-demo.md) | How in-browser inference works; design and accessibility |
| [docs/deployment.md](docs/deployment.md) | Docker, compose, CI, GitHub Pages |
| [docs/faq.md](docs/faq.md) | FAQ and limitations |

## ⚖️ Licence
Code and docs: MIT © 2026 Rishabh Aryan. Iris data: CC BY 4.0 — Fisher, R. A. (1936), UCI Machine Learning Repository,
doi:10.24432/C56C76.
