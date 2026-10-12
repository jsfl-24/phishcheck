# AGENTS.md

## Project

Phishing Website URL Detection — a Random Forest classifier that detects phishing URLs from URL-syntax features. Data comes from the Hugging Face dataset `pirocheto/phishing-url` (official train/test parquet splits). This is a small academic assignment: prioritize correctness, minimal code, and reproducibility over features.

## Hard rules

- One algorithm only: `RandomForestClassifier`. No deep learning, no model comparisons, no tuning searches.
- Use the official HF train/test parquet splits only. Never re-split, never cross-validate, never use the test split for design decisions.
- Never fabricate feature values. The model uses ONLY URL-syntax features that can be recomputed from a raw URL string with pure Python. Page-content features (hyperlinks, forms, iframe) and external-service features (WHOIS, DNS, page rank, traffic) are permanently out of scope.
- `random_state=42` for any stochastic component.
- Do not add Docker, databases, APIs, authentication, or deployment configs.
- Keep the root deliverables required by `initianPLAN.md`: `train_model.py`, `app.py`, `requirements.txt`, `README.md`, saved model, plots, report.

## Layout

- `train_model.py` (root) — load data, verify feature extraction, train, evaluate, save model + plots
- `app.py` (root) — minimal Streamlit UI for single-URL prediction
- `src/ml/url_features.py` — SINGLE source of truth for feature extraction; imported by both root scripts as `from ml.url_features import extract_features`
- `feature_columns.json` (generated) — canonical feature order; the app MUST build model input in this exact order
- `phishing_model.pkl` (generated) — trained model
- `outputs/` (generated) — `class_distribution.png`, `confusion_matrix.png`, `feature_importance.png`
- `requirements.txt` — GENERATED via `uv export`; do not hand-edit

## Commands

```bash
uv sync                                   # install deps + editable `ml` package
uv run python train_model.py              # train, evaluate, save artifacts
uv run streamlit run app.py               # launch the app
uv export --no-hashes --no-emit-project -o requirements.txt   # regenerate requirements.txt
```

## Invariants

- Any change to `src/ml/url_features.py` requires re-running `uv run python train_model.py`, which must pass the extraction verification harness (>=99% exact match vs stored dataset values on a ~500-row train sample). Features that cannot be reproduced are DROPPED, never approximated.
- The Streamlit app must not invent or default feature values; it predicts only from features extracted from the user-entered URL.
- README.md and the assignment report must state actual measured metrics; never invent numbers.
- Known sentinel values in the dataset (e.g. `domain_age = -1`) live only in out-of-scope columns; if any kept feature shows missing/sentinel values, impute with train-only statistics.
