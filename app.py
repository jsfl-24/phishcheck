import json
import pickle

import pandas as pd
import streamlit as st

from ml.url_features import JSON_PATH, extract_features

MODEL_PATH = "phishing_model.pkl"

st.set_page_config(page_title="Phishing URL Detection", layout="wide")


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


model = load_model()
feature_cols = json.loads(JSON_PATH.read_text(encoding="utf-8"))

st.title("Phishing Website URL Detection")
st.caption("Random Forest classifier — all features are computed from the URL string alone.")

url = st.text_input("Website URL", placeholder="https://example.com/login")
predict = st.button("Predict", type="primary")

if predict:
    if not url.strip():
        st.warning("Enter a URL first.")
    else:
        feats = extract_features(url.strip())
        X = pd.DataFrame([{c: feats[c] for c in feature_cols}])
        proba = dict(zip(model.classes_, model.predict_proba(X)[0]))
        pred = max(proba, key=proba.get)
        conf = proba[pred]

        if pred == "phishing":
            st.error(f"**Phishing** — confidence {conf:.1%}")
        else:
            st.success(f"**Legitimate** — confidence {conf:.1%}")

        with st.expander("Extracted features used for this prediction"):
            st.dataframe(
                pd.DataFrame(
                    [{"feature": c, "value": feats[c]} for c in feature_cols]
                ),
                hide_index=True,
                use_container_width=True,
            )

        st.bar_chart(pd.Series(proba, name="probability"))
