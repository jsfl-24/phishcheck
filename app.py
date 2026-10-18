import base64
import csv
import io
import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

from ml.url_features import JSON_PATH, extract_features

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "phishing_model.pkl"
OUTPUTS = ROOT / "outputs"

st.set_page_config(page_title="PhishCheck", layout="centered")


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


model = load_model()
feature_cols = json.loads(JSON_PATH.read_text(encoding="utf-8"))
assert model.n_features_in_ == len(feature_cols)


@st.cache_data
def chart_uris():
    return {
        name: base64.b64encode((OUTPUTS / f"{name}.png").read_bytes()).decode()
        for name in ("class_distribution", "confusion_matrix", "feature_importance")
        if (OUTPUTS / f"{name}.png").exists()
    }


@st.cache_data
def metrics_data():
    p = OUTPUTS / "metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def predict_url(raw):
    feats = extract_features(raw)
    X = pd.DataFrame([{c: feats[c] for c in feature_cols}])
    proba = dict(zip(model.classes_, model.predict_proba(X)[0]))
    pred = max(proba, key=proba.get)
    return feats, pred, float(proba[pred]), proba


def risk_signals(feats):
    out = []
    if feats["punycode"]:
        out.append("Uses special characters that can imitate real brands")
    if feats["shortening_service"]:
        out.append("Hidden behind a link shortener")
    if feats["nb_at"]:
        out.append("Contains '@' — the real address may be different")
    if feats["https_token"]:
        out.append("Mentions 'https' inside the domain — a common disguise trick")
    if feats["http_in_path"]:
        out.append("Hides an 'http' address inside the page path")
    if feats["nb_dslash"]:
        out.append("Contains '//' inside the path — may redirect elsewhere")
    if feats["port"]:
        out.append("Uses a non-standard port number")
    if feats["abnormal_subdomain"]:
        out.append("Unusual number of subdomains")
    if feats["phish_hints"]:
        out.append("Contains words often used in phishing (login, verify, account…)")
    if feats["suspecious_tld"]:
        out.append("Ends with a domain ending often abused by scammers")
    if feats["path_extension"]:
        out.append("Asks to open or download a file")
    if feats["tld_in_path"] or feats["tld_in_subdomain"]:
        out.append("Domain-looking text inside the address — may imitate a real site")
    if feats["length_url"] >= 75:
        out.append("The address is very long")
    if feats["length_hostname"] >= 30:
        out.append("The domain name is very long")
    if feats["ratio_digits_host"] >= 0.5:
        out.append("The domain contains many numbers")
    if feats["nb_hyphens"] >= 3:
        out.append("Many dashes in the domain")
    return out


def explain(feats, is_bad):
    if is_bad:
        return risk_signals(feats)[:4] or [
            "No single red flag — judged by the overall URL pattern"
        ]
    return ["No known trick patterns detected", "Looks like a normal website address"]


def _check():
    st.session_state.checked_url = st.session_state.url_input.strip()


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
    .gate-note {
        font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 11px;
        color: var(--sub); margin: 0.75rem 0 0; line-height: 1.5;
    }
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
        display: flex; justify-content: center; padding: 1rem 0;
        border-bottom: 1px solid var(--line); color: var(--sub);
    }
    .rail-tab svg { display: block; }
    .rail-body {
        padding: 0 1.1rem 1.5rem; overflow-y: auto; opacity: 0;
        transition: opacity 160ms ease;
    }
    .rail:hover .rail-body, .rail:focus-within .rail-body { opacity: 1; }
    .rail-head {
        display: block; font-family: 'JetBrains Mono', ui-monospace, monospace;
        font-size: 10px; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase;
        color: var(--sub); border-bottom: 1px solid var(--line);
        padding: 1rem 0 0.55rem; margin-bottom: 0.7rem;
    }
    .rail-verdict { font-size: 1.2rem; font-weight: 500; letter-spacing: -0.015em; line-height: 1.25; }
    .rail-verdict.bad { color: var(--red); }
    .rail-verdict.good { color: var(--green); }
    .rail-conf {
        font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 11px;
        letter-spacing: 0.05em; text-transform: uppercase; color: var(--sub); margin-top: 0.35rem;
    }
    .rail-why { list-style: none; margin: 0.7rem 0 0; padding: 0; display: grid; gap: 0.4rem; }
    .rail-why li {
        font-size: 0.8rem; color: var(--sub); line-height: 1.45;
        padding-left: 1rem; position: relative;
    }
    .rail-why li::before { content: "\2014"; position: absolute; left: 0; color: var(--ink); }
    .rail-empty {
        font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 11px;
        color: var(--sub); line-height: 1.6; padding: 0.2rem 0 0.4rem;
    }
    .rail-section + .rail-section { border-top: 1px solid var(--line); }
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

    .why { margin-top: 1.2rem; padding-top: 1rem; border-top: 1px solid var(--line); }
    .why-label {
        display: block; font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 10px;
        font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; color: var(--sub);
        margin-bottom: 0.55rem;
    }
    .why ul { list-style: none; margin: 0; padding: 0; display: grid; gap: 0.45rem; }
    .why li {
        font-size: 0.9rem; color: var(--sub); line-height: 1.45;
        padding-left: 1.1rem; position: relative;
    }
    .why li::before { content: "\2014"; position: absolute; left: 0; color: var(--ink); }

    .stButton button[kind="secondary"] {
        background: #ffffff; color: var(--ink); border: 1px solid var(--line);
        border-radius: 3px; font-family: 'JetBrains Mono', ui-monospace, monospace;
        font-weight: 500; font-size: 12px; letter-spacing: 0.06em; text-transform: uppercase;
        padding: 0.5rem 0.9rem; transition: background-color 140ms ease;
        touch-action: manipulation; -webkit-tap-highlight-color: transparent;
    }
    .stButton button[kind="secondary"]:hover { background: var(--paper); }
    .stButton button[kind="secondary"]:active { background: #ececea; }
    .stButton button[kind="secondary"]:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

    .batch-summary {
        font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 11px;
        font-weight: 500; letter-spacing: 0.05em; text-transform: uppercase;
        color: var(--sub); margin: 0.4rem 0 0.7rem;
    }
    .batch-summary b { color: var(--ink); font-weight: 500; }

    .metric-row {
        display: flex; justify-content: space-between; padding: 0.4rem 0;
        border-bottom: 1px solid var(--line);
        font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 11px;
        letter-spacing: 0.04em; text-transform: uppercase; color: var(--sub);
    }
    .metric-row b {
        color: var(--ink); font-weight: 500; font-variant-numeric: tabular-nums;
        letter-spacing: 0;
    }
    .cm {
        display: grid; grid-template-columns: 1.5fr 1fr 1fr; margin-top: 0.75rem;
        font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 10px;
        color: var(--sub);
    }
    .cm span { padding: 0.3rem 0.4rem; text-align: right; font-variant-numeric: tabular-nums; }
    .cm .cm-lbl { text-align: left; text-transform: uppercase; letter-spacing: 0.05em; }
    .cm .cm-h { text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.75; }
    .cm b { color: var(--ink); font-weight: 500; text-align: right; }

    .tips { margin-top: 2.75rem; }
    .tips-head {
        font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 11px;
        font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase;
        color: var(--sub); border-bottom: 1px solid var(--line);
        padding-bottom: 0.55rem; margin: 0 0 0.9rem 0; display: block;
    }
    .tips-line { font-size: 0.95rem; color: var(--ink); margin: 0 0 1rem 0; line-height: 1.5; }
    .tips ul { list-style: none; margin: 0; padding: 0; display: grid; grid-template-columns: 1fr 1fr; }
    .tips li {
        display: flex; gap: 0.7rem; padding: 0.75rem 0; line-height: 1.45;
        border-bottom: 1px solid var(--line); font-size: 0.875rem; color: var(--sub);
    }
    .tips li:nth-child(odd) { padding-right: 1rem; border-right: 1px solid var(--line); }
    .tips li:nth-child(even) { padding-left: 1rem; }
    .tip-num {
        font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 10px;
        font-weight: 500; color: var(--accent); padding-top: 0.2rem; flex-shrink: 0;
    }
    @media (max-width: 767px) {
        .tips ul { grid-template-columns: 1fr; }
        .tips li:nth-child(odd) { padding-right: 0; border-right: 0; }
        .tips li:nth-child(even) { padding-left: 0; }
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

st.markdown(
    """
    <div class="topbar">
        <span class="wordmark">PHISHCHECK</span>
        <span class="status"><span class="dot"></span>MODEL READY</span>
    </div>
    """,
    unsafe_allow_html=True,
)

@st.dialog("Are you a human?", width="small")
def human_check():
    st.markdown(
        '<p class="gate-note">Allow camera access and take one photo to unlock PhishCheck.</p>',
        unsafe_allow_html=True,
    )
    _shot = st.camera_input("Camera", key="gate_cam")
    if _shot is not None:
        from ml.face_check import annotate, detect_faces, prepare

        try:
            _img = prepare(_shot.getvalue())
        except Exception:
            st.warning("Unreadable frame — try again.")
        else:
            _boxes = detect_faces(_img)
            if _boxes:
                st.image(
                    annotate(_img, _boxes),
                    caption=f"{len(_boxes)} face(s) detected — welcome",
                )
                st.session_state["human_ok"] = True
                st.rerun()
            else:
                st.warning("No face detected — look straight at the camera.")
    if st.button("Skip", key="gate_skip"):
        st.session_state["human_ok"] = True
        st.rerun()


if not st.session_state.get("human_ok"):
    human_check()
    st.button("Open human check", key="gate_reopen")
    st.stop()

st.markdown(
    """
    <h1 class="hero">Is this URL <span class="safe">safe?</span></h1>
    <p class="hero-sub">Paste any website URL and get an instant verdict.</p>
    """,
    unsafe_allow_html=True,
)

url = st.text_input(
    "URL", placeholder="https://example.com/login", key="url_input", on_change=_check
)
predict = st.button("Check URL", type="primary", width="stretch")
if predict:
    st.session_state.checked_url = st.session_state.url_input.strip()

with st.expander("Check many URLs at once"):
    batch_text = st.text_area(
        "URLs", placeholder="https://first.example\nhttps://second.example", height=120
    )
    if st.button("Check all", key="batch_btn"):
        all_lines = [ln.strip() for ln in batch_text.splitlines() if ln.strip()]
        if not all_lines:
            st.markdown(
                '<div class="banner warn" role="status"><span>Paste at least one URL.</span></div>',
                unsafe_allow_html=True,
            )
        else:
            lines = all_lines[:100]
            Xb = pd.DataFrame(
                [{c: extract_features(u)[c] for c in feature_cols} for u in lines]
            )
            probas = model.predict_proba(Xb)
            preds = model.classes_[probas.argmax(axis=1)]
            best = probas.max(axis=1)
            rows = [
                {
                    "URL": u,
                    "Verdict": "Phishing" if p == "phishing" else "Safe",
                    "Confidence": f"{b:.0%}",
                }
                for u, p, b in zip(lines, preds, best)
            ]
            n_bad = sum(r["Verdict"] == "Phishing" for r in rows)
            extra = (
                f" &middot; showing first {len(lines)} of {len(all_lines)}"
                if len(all_lines) > len(lines)
                else ""
            )
            st.markdown(
                f'<div class="batch-summary" role="status"><b>{n_bad}</b> phishing &middot; '
                f"<b>{len(rows) - n_bad}</b> safe{extra}</div>",
                unsafe_allow_html=True,
            )
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            buf = io.StringIO()
            _w = csv.writer(buf)
            _w.writerow(["url", "verdict", "confidence"])
            for r in rows:
                _w.writerow([r["URL"], r["Verdict"], r["Confidence"]])
            st.download_button(
                "Download CSV", buf.getvalue(), "phishcheck_results.csv", "text/csv"
            )

checked = st.session_state.get("checked_url")
_result = None
if checked:
    feats, pred, conf, proba = predict_url(checked)
    is_bad = pred == "phishing"
    _result = {
        "feats": feats,
        "is_bad": is_bad,
        "verdict": "Likely phishing — be careful" if is_bad else "Looks safe",
        "conf": conf,
        "phish_pct": float(proba["phishing"]) * 100,
        "legit_pct": float(proba["legitimate"]) * 100,
        "why": explain(feats, is_bad),
    }

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
    if k in _charts
)
if _result:
    _rail_why = "".join(f"<li>{w}</li>" for w in _result["why"])
    _rail_cls = "bad" if _result["is_bad"] else "good"
    _last = (
        f'<span class="rail-head">Last check</span>'
        f'<div class="rail-verdict {_rail_cls}">{_result["verdict"]}</div>'
        f'<div class="rail-conf">{_result["conf"]:.0%} confident</div>'
        f'<ul class="rail-why">{_rail_why}</ul>'
    )
else:
    _last = (
        '<span class="rail-head">Last check</span>'
        '<div class="rail-empty">No URL checked yet.</div>'
    )
_m = metrics_data()
if _m:
    _tn, _fp = _m["confusion_matrix"][0]
    _fn, _tp = _m["confusion_matrix"][1]
    _metrics = (
        '<span class="rail-head">Model metrics</span>'
        f'<div class="metric-row"><span>Accuracy</span><b>{_m["accuracy"]:.1%}</b></div>'
        f'<div class="metric-row"><span>Precision</span><b>{_m["precision"]:.1%}</b></div>'
        f'<div class="metric-row"><span>Recall</span><b>{_m["recall"]:.1%}</b></div>'
        f'<div class="metric-row"><span>F1</span><b>{_m["f1"]:.1%}</b></div>'
        '<div class="cm">'
        '<span class="cm-h"></span><span class="cm-h">called safe</span><span class="cm-h">called phish</span>'
        f'<span class="cm-lbl">really safe</span><span><b>{_tn}</b></span><span><b>{_fp}</b></span>'
        f'<span class="cm-lbl">really phish</span><span><b>{_fn}</b></span><span><b>{_tp}</b></span>'
        "</div>"
    )
else:
    _metrics = ""
_sections = f'<div class="rail-section">{_last}</div>'
if _metrics:
    _sections += f'<div class="rail-section">{_metrics}</div>'
_sections += f'<div class="rail-section"><span class="rail-head">Training charts</span>{_figures}</div>'
st.markdown(
    f'<nav class="rail" aria-label="Last check, model metrics and charts" tabindex="0">'
    f'<span class="rail-tab" title="Charts"><svg width="18" height="18" viewBox="0 0 18 18" '
    f'aria-hidden="true" fill="currentColor"><rect x="2" y="9" width="3.5" height="7" rx="1"/>'
    f'<rect x="7.25" y="3" width="3.5" height="13" rx="1"/>'
    f'<rect x="12.5" y="6.5" width="3.5" height="9.5" rx="1"/></svg></span>'
    f'<div class="rail-body">{_sections}</div></nav>',
    unsafe_allow_html=True,
)

if checked is not None and not checked:
    st.markdown(
        '<div class="banner warn" role="status"><span>Enter a URL to check.</span></div>',
        unsafe_allow_html=True,
    )
elif _result:
    _why_items = "".join(f"<li>{w}</li>" for w in _result["why"])
    _cls = "bad" if _result["is_bad"] else "good"
    st.markdown(
        f"""
        <div class="banner {_cls}" role="status">
            <p class="verdict">{_result["verdict"]}</p>
            <p class="conf">{_result["conf"]:.0%} confident</p>
            <div class="bars">
                <div class="bar-row">
                    <span class="lbl">Phishing</span>
                    <div class="track"><div class="fill bad" style="width: {_result["phish_pct"]:.0f}%"></div></div>
                    <span class="pct">{_result["phish_pct"]:.0f}%</span>
                </div>
                <div class="bar-row">
                    <span class="lbl">Legitimate</span>
                    <div class="track"><div class="fill good" style="width: {_result["legit_pct"]:.0f}%"></div></div>
                    <span class="pct">{_result["legit_pct"]:.0f}%</span>
                </div>
            </div>
            <div class="why">
                <span class="why-label">Why</span>
                <ul>{_why_items}</ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Features used for this prediction"):
        st.dataframe(
            pd.DataFrame(
                [{"feature": c, "value": _result["feats"][c]} for c in feature_cols]
            ),
            hide_index=True,
            width="stretch",
        )

_tips_line = (
    "This URL looks risky — do not enter any credentials on it."
    if _result and _result["is_bad"]
    else "Good habits matter even on sites that look fine."
)
_TIPS = [
    "Never sign in or enter passwords from a link — open the site directly instead.",
    "Test a suspicious URL in an incognito or private window first.",
    "For a real test, use an isolated environment (VM or a spare browser profile).",
    "Check the address letter by letter — look-alike domains are the classic trick.",
    "Never download files (.exe, .zip) from URLs you don't trust.",
    "When in doubt, scan the URL with a multi-engine checker (e.g. VirusTotal) before opening.",
]
_tips_items = "".join(
    f'<li><span class="tip-num">{i:02d}</span><span>{t}</span></li>'
    for i, t in enumerate(_TIPS, 1)
)
st.markdown(
    f'<section class="tips">'
    f'<h2 class="tips-head">Safety tips</h2>'
    f'<p class="tips-line">{_tips_line}</p>'
    f"<ul>{_tips_items}</ul></section>",
    unsafe_allow_html=True,
)

if st.session_state.get("human_ok") and st.button(
    "Reset human check", key="reset_human"
):
    st.session_state["human_ok"] = False
    st.rerun()

st.markdown(
    """
    <div class="app-footer">
        <span>Random Forest &middot; 37 URL-syntax features</span>
        <a href="https://github.com/jsfl-24/phishcheck" target="_blank" rel="noopener">github.com/jsfl-24/phishcheck</a>
    </div>
    """,
    unsafe_allow_html=True,
)
