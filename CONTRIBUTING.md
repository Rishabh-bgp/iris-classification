# Contributing
1. Fork, branch, `pip install -r requirements-dev.txt`.
2. `python -m src.train && ruff check . && pytest -q` must pass.
3. If you change the model, re-run training so `web/model.json` stays in sync (a test checks parity), and re-execute
   the notebook so outputs match the docs.
4. Open a pull request. Contributions are released under the MIT licence.
