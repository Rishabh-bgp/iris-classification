"""Load the bundled UCI Iris dataset."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "iris.data"
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
SPECIES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]


def load_iris_df() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE, header=None, names=FEATURES + ["species"])
    return df.dropna()
