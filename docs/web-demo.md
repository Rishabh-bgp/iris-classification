# Web demo
Live: https://rishabh-bgp.github.io/iris-classification/

* **Inference:** `web/app.js` loads `model.json` and computes `z = (x − mean) / scale`, `s = coef · z + intercept`,
  `softmax(s)` — the same maths as scikit-learn's multinomial `LogisticRegression.predict_proba`.
* **UI:** four sliders (ranges = dataset min/max), example and random buttons, species cards with probability bars,
  a speech-bubble result from **Petalla**, an original chibi flower-fairy mascot drawn in inline SVG, and a canvas scatter
  plot of petal length vs width with your flower marked.
* **Theme:** pastel sakura palette (baby pink, hot pink, rose, lavender, lilac, peach, cream), falling petals, sparkles.
  All art is original.
* **Accessibility:** labelled sliders with live values, `aria-live` result, focus outlines, dark-on-light text, and
  `prefers-reduced-motion` turns off petals and animations.
* **Performance:** no frameworks, no external requests besides the two small JSON files.
