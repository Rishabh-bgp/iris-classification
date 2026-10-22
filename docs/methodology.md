# Methodology
* **Preprocessing:** `StandardScaler` inside every `Pipeline`, so scaling is learned on training folds only.
* **Model comparison:** majority baseline, logistic regression, LDA, KNN, SVM (RBF), Gaussian naive Bayes, decision
  tree, random forest (300 trees) — stratified 10-fold CV, `random_state = 42`.
* **Tuning:** `GridSearchCV` on the same folds — logistic regression `C ∈ {0.01, 0.1, 1, 10, 100}`; SVM `C × gamma`;
  KNN `k = 1…29` (odd).
* **Evaluation:** out-of-fold predictions (`cross_val_predict`) for the confusion matrix and classification report, plus a
  stratified 70/30 hold-out.
* **Interpretation:** pairplot, PCA (PC1 0.728 and PC2 0.230 of variance), logistic-regression coefficients,
  random-forest importances.
* **Why logistic regression is deployed:** it is among the best tuned models and its parameters (scaler mean/scale,
  coefficients, intercepts) export to a few numbers in JSON, so the browser can compute exactly the same softmax.
  Note: the selected `C = 100` is the edge of the grid.
