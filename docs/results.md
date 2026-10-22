# Results
All numbers are copied from the executed notebook `notebooks/iris_classification.ipynb` and `python -m src.train`.

## Training script (`src/train.py`)
Best C = 100; 10-fold CV accuracy 0.973 ± 0.053; accuracy on the full training data 0.987.

## Out-of-fold classification report (tuned logistic regression)
| Species | Precision | Recall | F1 |
|---|---|---|---|
| setosa | 1.000 | 1.000 | 1.000 |
| versicolor | 0.979 | 0.940 | 0.959 |
| virginica | 0.942 | 0.980 | 0.961 |
| **accuracy** | | | **0.973** |

## Hold-out (70/30, 45 flowers)
Accuracy 0.911; versicolor F1 0.882, virginica F1 0.846, setosa 1.000.

## Interpretation
Standardised coefficients — virginica: petal length +8.79, petal width +7.62; setosa: petal length −5.72, petal width
−5.32. Random-forest importances: petal length 0.455, petal width 0.417, sepal length 0.107, sepal width 0.021.
Setosa is perfectly separated; all errors are versicolor ↔ virginica. Model differences are within one CV standard deviation.

## Browser parity
On the deployed GitHub Pages site, `predictProba` in JavaScript matches scikit-learn on all 150 flowers (max |Δp| = 4.4e-16).
