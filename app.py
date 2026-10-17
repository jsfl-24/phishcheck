import base64
import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

from ml.url_features import JSON_PATH, extract_features

MODEL_PATH = "phishing_model.pkl"

st.set_page_config(page_title="Phishing URL Detection", layout="centered")


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


model = load_model()
feature_cols = json.loads(JSON_PATH.read_text(encoding="utf-8"))


@st.cache_data
def chart_uris():
    return {
        name: base64.b64encode((Path("outputs") / f"{name}.png").read_bytes()).decode()
        for name in ("class_distribution", "confusion_matrix", "feature_importance")
    }

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@400;500&family=Tangerine:wght@400;700&display=swap');

    :root {
        --ink: #0e1012;
        --sub: #4a4d55;
        --paper: #f5f5f3;
        --line: #d9dadd;
        --accent: #2f4cff;
        --red: #ee4234;
        --green: #047857;
    }

    html, body, .stApp {
        font-family: 'Space Grotesk', system-ui, -apple-system, 'Segoe UI', sans-serif;
        color: var(--ink);
    }

    .block-container { padding-top: 1.25rem; padding-bottom: 3rem; max-width: 760px; }
    @media (min-width: 768px) { .block-container { padding-left: 4rem; } }
    header[data-testid="stHeader"] { background: transparent; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    .topbar {
        display: flex; justify-content: space-between; align-items: center;
        border-bottom: 1px solid var(--line); padding: 0.9rem 0; margin-bottom: 2.75rem;
        font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 12px;
        letter-spacing: 0.05em;
    }
    .wordmark { font-weight: 500; color: var(--ink); }
    .status { color: var(--green); font-weight: 500; display: flex; align-items: center; gap: 0.45rem; }
    .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--green); display: inline-block; }

    h1.hero {
        font-size: clamp(2.4rem, 7vw, 3.5rem); font-weight: 500;
        letter-spacing: -0.015em; line-height: 1.1; color: var(--ink);
        margin: 0 0 0.9rem 0; text-wrap: balance;
    }
    .hero-sub { font-size: 1.05rem; color: var(--sub); margin: 0; line-height: 1.5; }
    .safe {
        font-family: 'Tangerine', cursive; color: var(--green); font-weight: 700;
        font-size: 1.35em; letter-spacing: 0;
    }

    .rail {
        position: fixed; left: 0; top: 0; bottom: 0; width: 44px; z-index: 9999;
        background: var(--paper); border-right: 1px solid var(--line);
        transition: width 180ms ease; overflow: hidden;
        display: flex; flex-direction: column;
    }
    .rail:hover, .rail:focus-within { width: 330px; }
    .rail-tab {
        align-self: center; writing-mode: vertical-rl; padding: 1.25rem 0.35rem;
        font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 11px;
        font-weight: 500; letter-spacing: 0.12em; text-transform: uppercase;
        color: var(--sub); border-bottom: 1px solid var(--line); white-space: nowrap;
    }
    .rail-body {
        padding: 0 1rem 1.5rem; overflow-y: auto; opacity: 0;
        transition: opacity 160ms ease;
    }
    .rail:hover .rail-body, .rail:focus-within .rail-body { opacity: 1; }
    .rail-figure { margin: 0 0 1.1rem; }
    .rail-figure img {
        width: 100%; display: block; border: 1px solid var(--line);
        border-radius: 3px; background: #ffffff;
    }
    .rail-figure figcaption {
        font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 10px;
        font-weight: 500; letter-spacing: 0.06em; text-transform: uppercase;
        color: var(--sub); margin-top: 0.4rem;
    }
    @media (max-width: 767px) { .rail { display: none; } }

    [data-testid="stTextInput"] { margin-bottom: 0.5rem; }
    [data-testid="stTextInput"] label {
        font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 11px;
        font-weight: 500; letter-spacing: 0.06em; text-transform: uppercase; color: var(--sub);
    }
    [data-testid="stTextInput"] input {
        background: transparent; border: 0; border-bottom: 1px solid var(--line);
        border-radius: 0; font-family: 'Space Grotesk', system-ui, sans-serif;
        font-size: 1.25rem; color: var(--ink); padding-left: 0;
        transition: border-color 140ms ease, box-shadow 140ms ease;
    }
    [data-testid="stTextInput"] input::placeholder {
        font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 0.95rem; color: #9a9da5;
    }
    [data-testid="stTextInput"] input:focus {
        border-bottom-color: var(--accent);
        box-shadow: 0 1px 0 0 var(--accent), 0 0 0 3px rgba(47, 76, 255, 0.14);
        outline: none;
    }

    .stButton button[kind="primary"] {
        background: var(--accent); color: #ffffff; border: 0; border-radius: 3px;
        font-family: 'JetBrains Mono', ui-monospace, monospace; font-weight: 500;
        font-size: 13px; letter-spacing: 0.06em; text-transform: uppercase;
        padding: 0.7rem 1rem; transition: background-color 140ms ease;
        touch-action: manipulation; -webkit-tap-highlight-color: transparent;
    }
    .stButton button[kind="primary"]:hover { background: #2440db; }
    .stButton button[kind="primary"]:active { background: #1d35b8; }
    .stButton button[kind="primary"]:focus-visible {
        outline: 2px solid var(--accent); outline-offset: 2px;
    }

    .banner {
        background: var(--paper); border: 1px solid var(--line); border-radius: 4px;
        padding: 1.5rem 1.6rem; margin: 0.5rem 0 0.2rem;
    }
    .banner.warn { padding: 0.85rem 1rem; }
    .banner.warn span {
        font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 12px;
        font-weight: 500; letter-spacing: 0.05em; text-transform: uppercase; color: var(--sub);
    }
    .verdict {
        font-size: 2.5rem; font-weight: 500; letter-spacing: -0.015em; line-height: 1.15; margin: 0;
    }
    .banner.bad .verdict { color: var(--red); }
    .banner.good .verdict { color: var(--green); }
    .conf {
        font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 12px;
        font-weight: 500; letter-spacing: 0.05em; text-transform: uppercase;
        color: var(--sub); margin: 0.4rem 0 0;
    }
    .bars { display: grid; gap: 0.7rem; margin-top: 1.3rem; }
    .bar-row {
        display: flex; align-items: center; gap: 0.9rem;
        font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 11px;
        letter-spacing: 0.05em; text-transform: uppercase; color: var(--sub);
    }
    .bar-row .lbl { width: 6rem; }
    .track { flex: 1; height: 6px; background: var(--line); border-radius: 2px; overflow: hidden; }
    .fill { height: 100%; border-radius: 2px; }
    .fill.bad { background: var(--red); }
    .fill.good { background: var(--green); }
    .pct {
        width: 3rem; text-align: right; font-weight: 500; color: var(--ink);
        font-variant-numeric: tabular-nums;
    }

    [data-testid="stExpander"] {
        background: #ffffff; border: 1px solid var(--line); border-radius: 4px;
    }
    [data-testid="stExpander"] summary {
        font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 12px;
        font-weight: 500; letter-spacing: 0.05em; text-transform: uppercase;
    }

    .app-footer {
        border-top: 1px solid var(--line); margin-top: 3rem; padding-top: 1.1rem;
        font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 12px;
        letter-spacing: 0.05em; color: var(--sub); display: flex; flex-wrap: wrap;
        justify-content: space-between; gap: 0.5rem;
    }
    .app-footer a { color: var(--accent); text-decoration: none; }
    .app-footer a:hover { color: var(--ink); }
    .app-footer a:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

    @media (prefers-reduced-motion: reduce) {
        * { transition: none !important; animation: none !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

_charts = chart_uris()
_CAPTIONS = {
    "class_distribution": "Class distribution",
    "confusion_matrix": "Confusion matrix",
    "feature_importance": "Feature importance",
}
_figures = "".join(
    f'<figure class="rail-figure">'
    f'<img src="data:image/png;base64,{_charts[k]}" alt="{_CAPTIONS[k]} chart" loading="lazy">'
    f"<figcaption>{_CAPTIONS[k]}</figcaption></figure>"
    for k in _CAPTIONS
)
st.markdown(
    f'<nav class="rail" aria-label="Model charts">'
    f'<span class="rail-tab">Charts</span>'
    f'<div class="rail-body">{_figures}</div></nav>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="topbar">
        <span class="wordmark">PHISHCHECK</span>
        <span class="status"><span class="dot"></span>MODEL READY</span>
    </div>
    <h1 class="hero">Is this URL <span class="safe">safe?</span></h1>
    <p class="hero-sub">Paste any website URL and get an instant verdict.</p>
    """,
    unsafe_allow_html=True,
)

url = st.text_input("URL", placeholder="https://example.com/login")
predict = st.button("Check URL", type="primary", width="stretch")

if predict:
    if not url.strip():
        st.markdown(
            '<div class="banner warn" role="status"><span>Enter a URL to check.</span></div>',
            unsafe_allow_html=True,
        )
    else:
        feats = extract_features(url.strip())
        X = pd.DataFrame([{c: feats[c] for c in feature_cols}])
        proba = dict(zip(model.classes_, model.predict_proba(X)[0]))
        pred = max(proba, key=proba.get)
        conf = proba[pred]
        is_bad = pred == "phishing"
        verdict = "Phishing" if is_bad else "Legitimate"
        phish_pct = float(proba["phishing"]) * 100
        legit_pct = float(proba["legitimate"]) * 100
        cls = "bad" if is_bad else "good"
        st.markdown(
            f"""
            <div class="banner {cls}" role="status">
                <p class="verdict">{verdict}</p>
                <p class="conf">Confidence &middot; {conf:.0%}</p>
                <div class="bars">
                    <div class="bar-row">
                        <span class="lbl">Phishing</span>
                        <div class="track"><div class="fill bad" style="width: {phish_pct:.0f}%"></div></div>
                        <span class="pct">{phish_pct:.0f}%</span>
                    </div>
                    <div class="bar-row">
                        <span class="lbl">Legitimate</span>
                        <div class="track"><div class="fill good" style="width: {legit_pct:.0f}%"></div></div>
                        <span class="pct">{legit_pct:.0f}%</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Features used for this prediction"):
            st.dataframe(
                pd.DataFrame(
                    [{"feature": c, "value": feats[c]} for c in feature_cols]
                ),
                hide_index=True,
                width="stretch",
            )

st.markdown(
    """
    <div class="app-footer">
        <span>Random Forest &middot; 37 URL-syntax features</span>
        <a href="https://github.com/jsfl-24/ML" target="_blank" rel="noopener">github.com/jsfl-24/ML</a>
    </div>
    """,
    unsafe_allow_html=True,
)
