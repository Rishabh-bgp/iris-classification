"""Train the Iris classifier and export it.

Outputs
* models/iris_model.joblib  - scikit-learn pipeline used by the API
* web/model.json            - plain parameters for in-browser inference
* models/metrics.json       - cross-validation and tuning results
"""
import json

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import FEATURES, ROOT, SPECIES, load_iris_df

SEED = 42


def build_pipeline(c: float = 1.0) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=c, max_iter=5000)),
    ])


def train() -> dict:
    df = load_iris_df()
    X, y = df[FEATURES].to_numpy(), df["species"].map(SPECIES.index).to_numpy()
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    grid = GridSearchCV(build_pipeline(), {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=cv, scoring="accuracy")
    grid.fit(X, y)
    best_c = grid.best_params_["clf__C"]
    scores = cross_val_score(build_pipeline(best_c), X, y, cv=cv, scoring="accuracy")
    model = build_pipeline(best_c).fit(X, y)

    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(model, ROOT / "models" / "iris_model.joblib")
    sc, clf = model.named_steps["scaler"], model.named_steps["clf"]
    params = {
        "features": FEATURES, "classes": SPECIES,
        "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
        "coef": clf.coef_.tolist(), "intercept": clf.intercept_.tolist(),
        "C": best_c,
    }
    (ROOT / "web" / "model.json").write_text(json.dumps(params, indent=1))
    metrics = {"best_C": best_c, "cv_accuracy_mean": float(np.mean(scores)),
               "cv_accuracy_std": float(np.std(scores)), "train_accuracy": float(model.score(X, y))}
    (ROOT / "models" / "metrics.json").write_text(json.dumps(metrics, indent=1))
    return metrics


if __name__ == "__main__":
    print(json.dumps(train(), indent=1))
