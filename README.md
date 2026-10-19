# PhishCheck

Phishing website URL detection using a Random Forest classifier. All 37 input features are URL-syntax features recomputed from the raw URL string with pure Python — no page fetching, no WHOIS/DNS, no network calls — each verified at ≥99% exact match against the stored values of the [`pirocheto/phishing-url`](https://huggingface.co/datasets/pirocheto/phishing-url) Hugging Face dataset before training. The official train (7,658 rows) / test (3,772 rows) splits are used as-is, with a single algorithm and `random_state=42`.

## Model performance

Measured on the official test split (3,772 rows), positive class = `phishing`:

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 0.8950 |
| Precision | 0.8872 |
| Recall    | 0.9051 |
| F1        | 0.8961 |

Confusion matrix (rows = actual, columns = predicted):

|                             | predicted legitimate | predicted phishing |
| --------------------------- | -------------------- | ------------------ |
| **actual legitimate**       | 1669                 | 217                |
| **actual phishing**         | 179                  | 1707               |

![Confusion matrix](outputs/confusion_matrix.png)

![Feature importance](outputs/feature_importance.png)

## Clone and run

```bash
git clone https://github.com/jsfl-24/phishcheck.git
cd phishcheck
uv sync
uv run streamlit run app.py
```

Then open http://localhost:8501, pass the face-based human check (take a photo with your camera, or use **Skip**), paste a URL, and get a verdict with confidence and the signals behind it. The trained model, feature schema, plots and metrics are committed, so the app runs right after `uv sync`.

## Face detection (human check)

PhishCheck opens with an **"Are you a human?" popup** — a small modal window, not a separate page. The rest of the app stays locked behind it until the check passes.

**How it works**

1. The browser shows a live camera preview in the popup (permission prompt appears once).
2. Taking a photo sends that single frame to the server, where OpenCV's Haar cascade (`haarcascade_frontalface_default.xml`) searches it for faces.
3. One or more faces found → every face is boxed on the snapshot and the app unlocks. No face found → a "No face detected" warning lets you retake the photo.

**Controls**

- **Skip** — enter without a photo (fallback if the camera is unavailable or blocked).
- **Open human check** — reopens the popup if it was dismissed.
- **Reset human check** (footer) — locks the app and asks again.
- The check passes once per browser session; a new tab or browser start always asks again.

**Good to know**

- Detection is fully offline (`opencv-python-headless`, code in `src/ml/face_check.py`) — no cloud APIs, the photo never leaves your machine. Phone photos are EXIF-rotation corrected and oversized frames are downscaled before scanning.
- The camera requires `localhost` or HTTPS — browsers block it on plain-IP hosts.
- Limitations: tuned for frontal faces in reasonable light; it is a human-presence check, not liveness detection — any photo containing a face passes.

## Retrain

```bash
uv run python train_model.py
```

Re-verifies all 37 features against the dataset (aborts if any falls below 99% or stops matching its verified variant), retrains the model, and regenerates `phishing_model.pkl`, the plots and `outputs/metrics.json`.
