# Deployment
* **Docker:** `docker build -t iris-api . && docker run -p 8000:8000 iris-api` (the image trains the model at build time).
* **docker-compose:** `docker compose up --build` → API on :8000, static demo via nginx on :8080.
* **CI** (`.github/workflows/ci.yml`): Python 3.12 and 3.13 → `ruff check .` → `python -m src.train` → `pytest`.
* **GitHub Pages:** served from the `gh-pages` branch, which holds a copy of `web/`. To update the live demo, copy `web/*` to
  the root of `gh-pages` and push.

The Docker files were written but not built in the authoring environment (no Docker available there).
