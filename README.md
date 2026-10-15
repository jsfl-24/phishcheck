# Phishing Website URL Detection Using Machine Learning

A beginner-friendly academic project that detects phishing websites from their **URL string alone** using a Random Forest classifier, and provides a small Streamlit web app for single-URL prediction.

The core idea: a URL can be inspected offline, without fetching the page, without WHOIS/DNS lookups, and without sending data anywhere. Every feature used by the model is recomputed from the raw URL text with pure Python, and every one of them has been **verified against the official dataset values (≥99% exact match)** before being accepted into the model.

## Project Status (temporary section — will be removed once development finishes)

| Stage | Deliverable                                                             | Status                                                                                                        |
| ----- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| 1     | `src/ml/url_features.py` — feature extraction + verification harness | **Done** — 37/37 features verified ≥99% on the official train split; `feature_columns.json` written |
| 2     | `train_model.py` — train Random Forest, evaluate, save model + plots | Not started                                                                                                   |
| 3     | `app.py` — Streamlit UI for single-URL prediction                    | Not started                                                                                                   |
| 4     | Report + final README polish                                            | README updated; assignment report pending                                                                     |

Notes on current state:

- No trained model yet, so there are **no accuracy numbers to report** — measured metrics will be added here after Stage 2 (never estimated or invented).
- Stage 1 detail: 13 URL-syntax features could not be reproduced at ≥99% fidelity and were **dropped**, not approximated. Page-content and external-service features are permanently out of scope.

## Tech Stack

| Layer                                | Choice                                                                                            |
| ------------------------------------ | ------------------------------------------------------------------------------------------------- |
| Language                             | Python ≥ 3.12                                                                                    |
| Environment / package manager        | [uv](https://docs.astral.sh/uv/) (`pyproject.toml`, `uv.lock`, generated `requirements.txt`) |
| Data handling                        | pandas + pyarrow (HF Parquet files)                                                               |
| Dataset hosting                      | Hugging Face Datasets —`pirocheto/phishing-url` via `huggingface_hub`                        |
| Model                                | scikit-learn`RandomForestClassifier` (`random_state=42`)                                      |
| Plots                                | matplotlib + seaborn                                                                              |
| Web app                              | Streamlit                                                                                         |
| Deep learning / tuning / comparisons | intentionally none — single algorithm, single train/test protocol                                |

## Dataset

| Property              | Value                                                                                                                                                                                                |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Source                | Hugging Face dataset[`pirocheto/phishing-url`](https://huggingface.co/datasets/pirocheto/phishing-url)                                                                                              |
| Splits used           | Official`train.parquet` (7,658 rows) and `test.parquet` (3,772 rows) — **never re-split, never cross-validated**                                                                          |
| Columns               | 89 total:`url` (raw string), `status` (target: `phishing` / `legitimate`), and 87 numeric feature columns                                                                                    |
| Target                | `status` — `phishing` vs `legitimate`, perfectly balanced (train: 3,829 / 3,829)                                                                                                              |
| Usage in this project | Features are recomputed from`url`; the stored feature columns are only used to **verify** the extraction code. The test split is used only for final evaluation, never for design decisions. |

The 87 stored feature columns break down exactly as:

1. **37 URL-syntax features used by this project** — computable from the URL text alone (verified, see below).
2. **13 URL-syntax features attempted and dropped** — could not be reproduced at ≥99% fidelity.
3. **37 features out of scope** — page-content features (hyperlinks, forms, iframe, titles, …), external-service features (WHOIS, DNS, page rank, traffic, google index, …), and a few columns that need the page or brand lists anyway (`nb_redirection`, `domain_in_brand`, `statistical_report`, …).

## Features

37 features are used, all recomputable from the URL string alone. Each one is a pure function of the raw URL, and each matched its stored dataset value on ≥99% of train rows (measured range 99.37%–100%, printed by the verification harness).

**Lengths & character counts (in the raw URL)**

`length_url`, `length_hostname`, `nb_dots`, `nb_hyphens`, `nb_at`, `nb_qm`, `nb_and`, `nb_or`, `nb_eq`, `nb_underscore`, `nb_tilde`, `nb_percent`, `nb_slash`, `nb_star`, `nb_colon`, `nb_comma`, `nb_semicolumn`, `nb_dollar`, `nb_space` (percent-decoded spaces), `nb_www`, `nb_com` (case-insensitive, adjusted for the host's last two labels), `nb_dslash`

**Scheme & host**

`https_token` (1 when the scheme is plain `http`), `http_in_path` (`http` token inside path/query), `port`, `punycode` (`xn--` in host), `ratio_digits_url`, `ratio_digits_host`

**Domain & TLD structure**

`tld_in_path` (host TLD reappears in path/query), `tld_in_subdomain` (TLD label inside the subdomain region), `abnormal_subdomain` (numeric/odd prefixes in subdomains or IP host), `suspecious_tld` (suspicious TLD, with second-level context such as `go.id` / `gob.pe`), `shortening_service` (known URL-shortener host)

**Path & words**

`path_extension` (last path segment ends in `.txt`/`.csv`/`.log`), `longest_words_raw` (longest token in the scheme-less URL), `longest_word_host` (longest token in the host), `phish_hints` (count of hint words such as `login`, `admin`, `verify`, `paypal`)

The full canonical order lives in `feature_columns.json` (generated by the harness); the training script and the app must build the model input in exactly this order.

**Excluded on purpose**

- 37 stored columns (page-content / external-service / redirect & brand columns) — cannot be recomputed from a URL offline.
- The 13 URL-syntax candidates listed above — attempted and dropped because they could not be reproduced at ≥99%: `ip`, `prefix_suffix`, `nb_subdomains`, `char_repeat`, `random_domain`, `length_words_raw`, `shortest_words_raw`, `avg_words_raw`, `shortest_word_host`, `avg_word_host`, `shortest_word_path`, `longest_word_path`, `avg_word_path`.

## How to Use

Everything runs through `uv` from the project root (`E:\Projects\ML`).

**1. Install dependencies**

```bash
uv sync
```

**2. Verify the feature extraction (Stage 1 deliverable)**

```bash
uv run python -m ml.url_features
```

Prints a per-feature table (feature, best variant, exact-match rate on the 7,658 train rows, KEEP/DROP verdict) and (re)writes `feature_columns.json`. Any feature below 99% must be fixed or dropped — never approximated.

**3. Train the model** *(Stage 2 — not implemented yet)*

```bash
uv run python train_model.py
```

Will train a `RandomForestClassifier` (`random_state=42`) on the official train split, evaluate on the official test split, save `phishing_model.pkl` and the plots (`outputs/class_distribution.png`, `outputs/confusion_matrix.png`, `outputs/feature_importance.png`), and print the same verification table.

**4. Launch the web app** *(Stage 3 — not implemented yet)*

```bash
uv run streamlit run app.py
```

Paste any URL; the app extracts the 37 features from your input (no fabricated values) and shows the model's prediction.

**5. Regenerate `requirements.txt`** (only if dependencies change)

```bash
uv export --no-hashes --no-emit-project -o requirements.txt
```

## Project Layout

```
train_model.py              # Stage 2: train / evaluate / save artifacts (pending)
app.py                      # Stage 3: Streamlit prediction UI (pending)
src/ml/url_features.py      # feature extraction + verification harness (single source of truth)
feature_columns.json        # generated canonical feature order
phishing_model.pkl          # generated trained model (pending)
outputs/                    # generated plots (pending)
requirements.txt            # generated via uv export
AGENTS.md                   # development rules for this repository
initianPLAN.md              # assignment plan / spec
```
