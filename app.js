"use strict";
// In-browser inference: standardise -> linear scores -> softmax (same maths as scikit-learn's LogisticRegression).
const META = {
  "Iris-setosa": {short: "setosa", emoji: "🌷", color: "#e75480", line: "Setosa! Tiny petals, big heart 🌷"},
  "Iris-versicolor": {short: "versicolor", emoji: "💜", color: "#8a5cd6", line: "Versicolor! Lovely lavender vibes 💜"},
  "Iris-virginica": {short: "virginica", emoji: "🍑", color: "#e8893c", line: "Virginica! So tall and elegant 🍑"},
};
const LABELS = {sepal_length: "Sepal length", sepal_width: "Sepal width", petal_length: "Petal length", petal_width: "Petal width"};
const PRESETS = {setosa: [5.1, 3.5, 1.4, 0.2], versicolor: [6.4, 3.2, 4.5, 1.5], virginica: [6.7, 3.0, 5.8, 2.2]};
let model, data, ranges;

function predictProba(x) {
  const z = x.map((v, i) => (v - model.mean[i]) / model.scale[i]);
  const s = model.coef.map((row, k) => row.reduce((a, w, i) => a + w * z[i], model.intercept[k]));
  const m = Math.max(...s), e = s.map(v => Math.exp(v - m)), t = e.reduce((a, b) => a + b, 0);
  return e.map(v => v / t);
}

function values() { return model.features.map(f => parseFloat(document.getElementById(f).value)); }

function render() {
  const x = values(), p = predictProba(x), best = p.indexOf(Math.max(...p));
  model.features.forEach((f, i) => { document.getElementById(f + "-out").textContent = x[i].toFixed(1); });
  model.classes.forEach((c, k) => {
    const el = document.getElementById("sp-" + k);
    el.classList.toggle("win", k === best);
    el.querySelector(".bar span").style.width = (p[k] * 100).toFixed(1) + "%";
    el.querySelector(".pct").textContent = (p[k] * 100).toFixed(1) + "%";
    el.setAttribute("aria-label", `${META[c].short}: ${(p[k] * 100).toFixed(1)} percent`);
  });
  const c = model.classes[best], conf = p[best];
  const bubble = document.getElementById("bubble");
  bubble.textContent = conf > 0.85 ? META[c].line : `Hmm… maybe ${META[c].short}? (${(conf * 100).toFixed(0)}% sure) 🤔`;
  const mascot = document.getElementById("mascot");
  document.getElementById("mouth").setAttribute("d", conf > 0.85 ? "M88 130 Q100 146 112 130" : "M90 136 Q100 132 110 136");
  mascot.classList.remove("bounce"); void mascot.offsetWidth; mascot.classList.add("bounce");
  drawScatter(x);
}

function drawScatter(x) {
  const cv = document.getElementById("scatter"), ctx = cv.getContext("2d"), W = cv.width, H = cv.height, P = 46;
  const xr = [0.5, 7.5], yr = [0, 2.8];
  const sx = v => P + (v - xr[0]) / (xr[1] - xr[0]) * (W - P * 1.5), sy = v => H - P - (v - yr[0]) / (yr[1] - yr[0]) * (H - P * 1.5);
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = "#f0c8da"; ctx.fillStyle = "#6e4560"; ctx.font = "15px sans-serif";
  for (let v = 1; v <= 7; v++) { ctx.beginPath(); ctx.moveTo(sx(v), sy(0)); ctx.lineTo(sx(v), sy(2.8)); ctx.stroke(); ctx.fillText(v, sx(v) - 3, H - P + 16); }
  for (let v = 0; v <= 2.5; v += 0.5) { ctx.beginPath(); ctx.moveTo(sx(0.5), sy(v)); ctx.lineTo(sx(7.5), sy(v)); ctx.stroke(); ctx.fillText(v.toFixed(1), 10, sy(v) + 4); }
  ctx.fillText("petal length (cm)", W / 2 - 50, H - 8);
  ctx.save(); ctx.translate(14, H / 2 + 40); ctx.rotate(-Math.PI / 2); ctx.fillText("petal width (cm)", 0, 0); ctx.restore();
  data.forEach(r => { ctx.globalAlpha = .7; ctx.fillStyle = META[r[4]].color; ctx.beginPath(); ctx.arc(sx(r[2]), sy(r[3]), 5, 0, 7); ctx.fill(); });
  ctx.globalAlpha = 1; ctx.font = "26px sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillText("💗", sx(x[2]), sy(x[3])); ctx.textAlign = "start"; ctx.textBaseline = "alphabetic";
}

function build() {
  ranges = model.features.map((_, i) => [Math.min(...data.map(r => r[i])), Math.max(...data.map(r => r[i]))]);
  const box = document.getElementById("sliders");
  model.features.forEach((f, i) => {
    const d = document.createElement("div"); d.className = "slider";
    d.innerHTML = `<label for="${f}">${LABELS[f]} <output id="${f}-out" for="${f}"></output></label>
      <input type="range" id="${f}" min="${ranges[i][0]}" max="${ranges[i][1]}" step="0.1" value="${PRESETS.setosa[i]}">`;
    box.appendChild(d); d.querySelector("input").addEventListener("input", render);
  });
  const cards = document.getElementById("cards");
  model.classes.forEach((c, k) => {
    cards.insertAdjacentHTML("beforeend", `<div class="sp" id="sp-${k}" role="img"><div class="emoji">${META[c].emoji}</div>
      <div class="name">${META[c].short}</div><div class="bar"><span></span></div><div class="pct"></div></div>`);
  });
  document.querySelectorAll(".presets button").forEach(b => b.addEventListener("click", () => {
    const v = b.dataset.p === "random" ? ranges.map(([a, z]) => Math.round((a + Math.random() * (z - a)) * 10) / 10) : PRESETS[b.dataset.p];
    model.features.forEach((f, i) => { document.getElementById(f).value = v[i]; }); render();
  }));
  if (!matchMedia("(prefers-reduced-motion: reduce)").matches) {
    const pet = document.querySelector(".petals");
    for (let i = 0; i < 18; i++) {
      const s = document.createElement("span"); s.className = "petal";
      s.style.left = Math.random() * 100 + "vw"; s.style.animationDuration = 7 + Math.random() * 8 + "s";
      s.style.animationDelay = -Math.random() * 12 + "s"; s.style.transform = `scale(${0.6 + Math.random()})`; pet.appendChild(s);
    }
  }
  render();
}

Promise.all([fetch("model.json").then(r => r.json()), fetch("iris.json").then(r => r.json())])
  .then(([m, d]) => { model = m; data = d; build(); });
