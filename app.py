"""Streamlit app: predict thyroid cancer recurrence from clinical features.

Trains a Random Forest on Thyroid_Diff.csv at startup (cached), so no model file is needed.
"""
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_FILE = "Thyroid_Diff.csv"
TARGET = "Recurred"
NUMERIC = ["Age"]

st.set_page_config(page_title="Thyroid Cancer Recurrence Predictor", page_icon="🩺")


def make_pipeline(categorical):
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", NUMERIC),
        ]
    )
    rf = RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=4, random_state=42)
    return Pipeline([("pre", pre), ("rf", rf)])


@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)


@st.cache_resource
def train(df):
    features = [c for c in df.columns if c != TARGET]
    categorical = [c for c in features if c not in NUMERIC]
    X, y = df[features], (df[TARGET] == "Yes").astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    model = make_pipeline(categorical).fit(X_tr, y_tr)
    metrics = {
        "Train accuracy": model.score(X_tr, y_tr),
        "Test accuracy": model.score(X_te, y_te),
        "5-fold CV": cross_val_score(make_pipeline(categorical), X, y, cv=5).mean(),
    }
    return model, metrics, features


st.title("🩺 Thyroid Cancer Recurrence Predictor")
st.caption(
    "Random Forest trained on the Differentiated Thyroid Cancer dataset. "
    "Educational portfolio project, not a medical device."
)

df = load_data()
model, metrics, features = train(df)

cols = st.columns(3)
for col, (name, value) in zip(cols, metrics.items()):
    col.metric(name, f"{value:.1%}")

st.subheader("Patient details")
inputs = {}
grid = st.columns(2)
for i, feat in enumerate(features):
    target_col = grid[i % 2]
    if feat in NUMERIC:
        inputs[feat] = target_col.slider(
            feat, int(df[feat].min()), int(df[feat].max()), int(df[feat].median())
        )
    else:
        inputs[feat] = target_col.selectbox(feat, sorted(df[feat].dropna().unique()))

if st.button("Predict recurrence", type="primary"):
    prob = float(model.predict_proba(pd.DataFrame([inputs])[features])[0][1])
    if prob >= 0.5:
        st.error(f"Higher chance of recurrence ({prob:.1%})")
    else:
        st.success(f"Lower chance of recurrence ({prob:.1%})")
    st.progress(prob)
    st.caption("Risk and Response are post-treatment factors and strongly drive this prediction.")
