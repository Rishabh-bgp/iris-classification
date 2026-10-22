# FAQ and limitations
**Why not the single best CV model?** LDA, tuned KNN and tuned logistic regression all reach 0.973; logistic
regression is used because it exports exactly to the browser.

**Why does the demo often show 100%?** With C = 100 the model is weakly regularised and very confident far from class
boundaries; probabilities near the versicolor/virginica boundary are more informative.

**Limitations:** 150 flowers, one historic dataset; scores are near the ceiling and differences between models are not
significant. Educational project.
