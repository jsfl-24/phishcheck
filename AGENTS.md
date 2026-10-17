
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

## Efficiency (tokens & speed)

- Run each command at most once per change; never re-run unchanged steps. Prefer reading over running (read > one run > re-run only after an edit).
- Read a file once, edit surgically, batch independent reads/commands in one message; never dump whole files, datasets, or tables — query only what is needed.
- The harness table from `uv run python -m ml.url_features` is the only verification evidence needed; print per-row failures only when a feature is <99%.
- CPU-local only: RandomForest with `random_state=42`, `n_jobs=-1`, sklearn defaults — training is seconds; no repeated runs or long monitoring loops.
- Keep code minimal: no comments, no docstrings, no refactors of passing code, no new dependencies. Prefer deleting code over adding it.
- Diagnostic scripts live in %TEMP%, never in the repo. Kill background processes by tree (`taskkill /T /F`); app smoke tests max ~10 s; `streamlit run` never exits on its own.

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

- Any change to `src/ml/url_features.py` requires re-running `uv run python train_model.py`, which must pass the extraction verification harness (>=99% exact match vs stored dataset values on the full 7,658-row train split). Features that cannot be reproduced are DROPPED, never approximated.
- The Streamlit app must not invent or default feature values; it predicts only from features extracted from the user-entered URL.
- README.md and the assignment report must state actual measured metrics; never invent numbers.
- Known sentinel values in the dataset (e.g. `domain_age = -1`) live only in out-of-scope columns; if any kept feature shows missing/sentinel values, impute with train-only statistics.

## Frontend design workflow (OpenCode skills)

Globally installed skills, connected to this project for all UI/visual work. They are NOT auto-loaded: before UI work, read the skill's `SKILL.md` — invoke it by exact ID when the session lists it, otherwise read it directly from `~/.agents/skills/<id>/SKILL.md`.

1. `frontend-design` — design & implementation. Use when creating or substantially redesigning UI. Establish a clear visual direction BEFORE writing UI code; produce distinctive, polished interfaces (typography, spacing, composition, color, visual hierarchy, responsive behavior, meaningful interactions) instead of generic AI layouts; use the existing framework, dependencies, and conventions; no unnecessary dependencies or unrelated rewrites.
2. `design-taste-frontend` — visual quality. Use to guide visual decisions: coherent design language for the product/audience; consistent spacing, typography, color, borders, shadows, component proportions; avoid gratuitous gradients, glassmorphism, generic card grids, and decorative clichés unless they genuinely fit; purposeful animations respecting reduced-motion; prioritize composition and consistency.
3. `web-design-guidelines` — design review. Use to review completed or modified UI: accessibility, semantic markup, keyboard navigation, focus states, contrast, responsive behavior, interaction/loading/error states; follow the skill's actual guidelines and review procedure; fix the relevant issues found.

Mandatory rules:

- New frontend feature: use `frontend-design` + `design-taste-frontend` during design and implementation, then `web-design-guidelines` for review. Small UI change: invoke only the relevant skills, still ensuring accessibility and visual consistency. Redesign: inspect the current implementation before changing it.
- Skills never override project architecture or hard rules: UI stays plain Streamlit, no new UI dependencies or JS build tooling, interface stays minimal, and backend/model/feature logic stays untouched unless explicitly required.
- Never fabricate skill instructions or claim a skill was used if its file was not actually read.
- Prefer reusable components, maintainable code, and consistent design tokens over duplicated styles.
- Do not stop at a plan: when asked to implement UI, make the requested changes and run the available checks.
