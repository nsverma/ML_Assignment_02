from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

from train_models import build_models

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

st.set_page_config(
    page_title="Diagnostic Classification Lab",
    page_icon="🧪",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    .main-title {font-size: 2.15rem; font-weight: 750; margin-bottom: .15rem;}
    .subtitle {color: #556070; margin-bottom: 1.3rem;}
    .info-box {padding: .85rem 1rem; border: 1px solid #dfe5ec; border-radius: 12px; background: #f7f9fc;}
    div[data-testid="stMetric"] {border: 1px solid #e2e8f0; padding: .7rem; border-radius: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_assets():
    metadata = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("sklearn_version") == sklearn_version:
        models = {
            name: joblib.load(MODEL_DIR / filename)
            for name, filename in metadata["model_files"].items()
        }
        return metadata, models

    # Pickled scikit-learn estimators are not guaranteed to work across versions.
    # Rebuild equivalent in-memory models when the runtime differs from training.
    df = pd.read_csv(ROOT / "full_dataset.csv")
    features = metadata["feature_columns"]
    target = metadata["target"]
    X_train, _, y_train, _ = train_test_split(
        df[features],
        df[target],
        test_size=0.20,
        random_state=metadata["random_state"],
        stratify=df[target],
    )
    models = build_models()
    for model in models.values():
        model.fit(X_train, y_train)
    return metadata, models


def get_metrics(model, X, y):
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]
    return {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, prob),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }, pred, prob


metadata, models = load_assets()
features = metadata["feature_columns"]

st.markdown('<div class="main-title">Diagnostic Classification Lab</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Machine Learning Assignment 2 · Breast Cancer Wisconsin (Diagnostic)</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Experiment Controls")
    uploaded = st.file_uploader(
        "Upload test data (CSV)",
        type=["csv"],
        help="The CSV must contain all 30 feature columns and the diagnosis target column.",
    )
    selected_model = st.selectbox("Select a model", list(models.keys()))
    st.divider()
    st.caption("Target mapping")
    st.write("**1 = malignant**")
    st.write("**0 = benign**")
    st.caption("Split used during training: 80% train / 20% test, stratified, random_state=42")

if uploaded is None:
    df = pd.read_csv(ROOT / "test_data.csv")
    st.info("Using the repository's packaged test_data.csv. Upload another compatible test CSV from the sidebar to re-evaluate the models.")
else:
    df = pd.read_csv(uploaded)
    st.success(f"Loaded uploaded test file with {len(df)} rows.")

required = features + [metadata["target"]]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error("The uploaded CSV is missing required columns: " + ", ".join(missing))
    st.stop()

try:
    X = df[features].apply(pd.to_numeric, errors="raise")
    y = pd.to_numeric(df[metadata["target"]], errors="raise").astype(int)
except Exception as exc:
    st.error(f"All feature and target values must be numeric. Details: {exc}")
    st.stop()

if not set(y.unique()).issubset({0, 1}):
    st.error("The diagnosis column must contain only 0 (benign) and 1 (malignant).")
    st.stop()

# Overall comparison on current test data.
comparison_rows = []
cache = {}
for name, model in models.items():
    try:
        m, pred, prob = get_metrics(model, X, y)
    except ValueError as exc:
        st.error(f"Could not evaluate {name}: {exc}")
        st.stop()
    cache[name] = (m, pred, prob)
    comparison_rows.append({"ML Model Name": name, **m})

comparison = pd.DataFrame(comparison_rows)
comparison = comparison.sort_values(["MCC", "F1", "AUC"], ascending=False).reset_index(drop=True)

left, right = st.columns([1.25, 1])
with left:
    st.subheader("Dataset Snapshot")
    st.dataframe(df.head(10), use_container_width=True)
with right:
    st.subheader("Dataset Summary")
    c1, c2 = st.columns(2)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Features", len(features))
    counts = y.value_counts().to_dict()
    c3, c4 = st.columns(2)
    c3.metric("Malignant", int(counts.get(1, 0)))
    c4.metric("Benign", int(counts.get(0, 0)))
    st.markdown(
        f'<div class="info-box"><b>Source:</b> {metadata["dataset_name"]}<br><b>Positive class:</b> malignant (1)</div>',
        unsafe_allow_html=True,
    )

st.subheader("All-Model Comparison")
st.dataframe(
    comparison.style.format({c: "{:.4f}" for c in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]}),
    use_container_width=True,
    hide_index=True,
)

st.subheader(f"Selected Model: {selected_model}")
metrics, predictions, probabilities = cache[selected_model]
metric_cols = st.columns(6)
for col, key in zip(metric_cols, ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]):
    col.metric(key, f"{metrics[key]:.4f}")

cm = confusion_matrix(y, predictions, labels=[0, 1])
plot_col, report_col = st.columns([1, 1.25])
with plot_col:
    st.markdown("#### Confusion Matrix")
    fig, ax = plt.subplots(figsize=(4.4, 3.7))
    image = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Benign (0)", "Malignant (1)"],
        yticklabels=["Benign (0)", "Malignant (1)"],
        ylabel="Actual",
        xlabel="Predicted",
    )
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

with report_col:
    st.markdown("#### Classification Report")
    report = classification_report(
        y,
        predictions,
        labels=[0, 1],
        target_names=["Benign", "Malignant"],
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

st.markdown("#### Prediction Preview")
preview = pd.DataFrame({
    "Actual": y.values,
    "Predicted": predictions,
    "Malignant probability": probabilities,
})
preview["Actual label"] = preview["Actual"].map({0: "Benign", 1: "Malignant"})
preview["Predicted label"] = preview["Predicted"].map({0: "Benign", 1: "Malignant"})
st.dataframe(preview.head(25), use_container_width=True, hide_index=True)

winner = comparison.iloc[0]
st.caption(
    f"Current test-data winner by MCC (tie-breakers: F1 then AUC): {winner['ML Model Name']} · MCC {winner['MCC']:.4f}."
)
