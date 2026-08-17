from pathlib import Path
import json
import joblib
import pandas as pd
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
    page_title="Breast cancer model audit",
    page_icon=":material/health_metrics:",
    layout="wide",
)

MODEL_NOTES = {
    "Logistic Regression": "A scaled linear baseline that estimates malignancy probability from all 30 measurements.",
    "Decision Tree": "A compact rule-based model limited to depth 5 to reduce overfitting.",
    "kNN": "A scaled neighborhood model that votes using the seven most similar training cases.",
    "Naive Bayes": "A probabilistic baseline that assumes the measurements are conditionally independent.",
    "Random Forest (Ensemble)": "An ensemble of 400 trees designed to reduce the instability of a single tree.",
}

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

st.title("Breast cancer model audit")
st.caption(
    "A reproducible comparison of five classifiers on the Breast Cancer "
    "Wisconsin (Diagnostic) dataset. This educational app is not a medical device."
)
st.markdown(":blue-badge[30 measurements] :violet-badge[5 models] :green-badge[Malignant = 1]")

with st.sidebar:
    st.header("Evaluation setup")
    uploaded = st.file_uploader(
        "Test dataset",
        type=["csv"],
        help="Upload a labeled CSV containing the 30 measurement columns and diagnosis.",
    )
    selected_model = st.selectbox(
        "Model to audit",
        list(models.keys()),
        help="The detailed error audit below follows this model.",
    )
    st.caption("Training protocol: stratified 80/20 split · random state 42")
    with st.expander("Expected CSV contract", icon=":material/table_chart:"):
        st.write("Thirty numeric feature columns plus `diagnosis`.")
        st.write("`0` = benign · `1` = malignant")

try:
    if uploaded is None:
        df = pd.read_csv(ROOT / "test_data.csv")
        data_source = "Packaged holdout set"
    else:
        df = pd.read_csv(uploaded)
        data_source = uploaded.name
except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
    st.error(f"The CSV could not be read: {exc}", icon=":material/error:")
    st.stop()

required = features + [metadata["target"]]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(
        "The CSV is missing required columns: " + ", ".join(missing),
        icon=":material/error:",
    )
    st.stop()

try:
    X = df[features].apply(pd.to_numeric, errors="raise")
    y_numeric = pd.to_numeric(df[metadata["target"]], errors="raise")
except (TypeError, ValueError) as exc:
    st.error(f"Feature and diagnosis values must be numeric: {exc}", icon=":material/error:")
    st.stop()

if set(y_numeric.unique()) != {0, 1}:
    st.error(
        "Evaluation requires both diagnosis classes, encoded exactly as 0 and 1.",
        icon=":material/error:",
    )
    st.stop()
y = y_numeric.astype(int)

comparison_rows = []
cache = {}
for name, model in models.items():
    try:
        m, pred, prob = get_metrics(model, X, y)
    except ValueError as exc:
        st.error(f"Could not evaluate {name}: {exc}")
        st.stop()
    model_cm = confusion_matrix(y, pred, labels=[0, 1])
    cache[name] = (m, pred, prob, model_cm)
    comparison_rows.append({
        "Model": name,
        **m,
        "False negatives": int(model_cm[1, 0]),
    })

comparison = pd.DataFrame(comparison_rows)
comparison = comparison.sort_values(["MCC", "F1", "AUC"], ascending=False).reset_index(drop=True)
comparison.insert(0, "Rank", comparison.index + 1)
winner = comparison.iloc[0]

counts = y.value_counts().to_dict()
summary_cols = st.columns(4)
summary_cols[0].metric("Cases evaluated", f"{len(df):,}")
summary_cols[1].metric("Malignant cases", int(counts.get(1, 0)))
summary_cols[2].metric("Benign cases", int(counts.get(0, 0)))
summary_cols[3].metric("MCC leader", winner["Model"])
st.caption(f"Evaluation source: {data_source}")

st.header("Selected model audit")
metrics, predictions, probabilities, cm = cache[selected_model]
with st.container(border=True):
    st.subheader(selected_model)
    st.write(MODEL_NOTES[selected_model])
    metric_cols = st.columns(4)
    metric_cols[0].metric("Malignant recall", f"{metrics['Recall']:.3f}")
    metric_cols[1].metric("Malignant precision", f"{metrics['Precision']:.3f}")
    metric_cols[2].metric("MCC", f"{metrics['MCC']:.3f}")
    metric_cols[3].metric("AUC", f"{metrics['AUC']:.3f}")

tn, fp, fn, tp = cm.ravel()
audit_left, audit_right = st.columns([1, 1.35])
with audit_left:
    st.subheader("Clinical error check")
    error_cols = st.columns(2)
    error_cols[0].metric("Missed malignant", int(fn), help="False negatives")
    error_cols[1].metric("Benign flagged", int(fp), help="False positives")
    if fn == 0:
        st.success("No malignant cases were missed in this test set.", icon=":material/check_circle:")
    else:
        st.warning(
            f"{int(fn)} malignant case(s) were predicted as benign.",
            icon=":material/warning:",
        )
    st.caption("False negatives are highlighted because missing a malignant case is the higher-risk error.")

with audit_right:
    st.subheader("Confusion matrix")
    cm_df = pd.DataFrame(
        cm,
        index=["Actual benign", "Actual malignant"],
        columns=["Predicted benign", "Predicted malignant"],
    )
    st.dataframe(cm_df, width="stretch")
    outcome_cols = st.columns(2)
    outcome_cols[0].metric("Correct benign", int(tn))
    outcome_cols[1].metric("Correct malignant", int(tp))

st.header("Model leaderboard")
st.caption("Models are ranked by MCC, then F1 and AUC. Lower false negatives are clinically preferable.")
st.dataframe(
    comparison.style.format({c: "{:.4f}" for c in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]}),
    width="stretch",
    hide_index=True,
)

with st.expander("Inspect class-level report", icon=":material/analytics:"):
    report = classification_report(
        y,
        predictions,
        labels=[0, 1],
        target_names=["Benign", "Malignant"],
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.4f}"), width="stretch")

preview = pd.DataFrame({
    "Actual diagnosis": y.map({0: "Benign", 1: "Malignant"}).values,
    "Model decision": pd.Series(predictions).map({0: "Benign", 1: "Malignant"}),
    "Malignancy probability": probabilities,
})
preview["Outcome"] = preview.apply(
    lambda row: "Correct" if row["Actual diagnosis"] == row["Model decision"] else "Error",
    axis=1,
)

with st.expander("Review individual predictions", icon=":material/search:"):
    only_errors = st.toggle("Show errors only")
    visible_preview = preview[preview["Outcome"] == "Error"] if only_errors else preview
    st.dataframe(
        visible_preview,
        width="stretch",
        hide_index=True,
        column_config={
            "Malignancy probability": st.column_config.ProgressColumn(
                "Malignancy probability",
                min_value=0.0,
                max_value=1.0,
                format="%.3f",
            )
        },
    )

with st.expander("Inspect input data", icon=":material/table_chart:"):
    st.caption(f"Showing the first 10 of {len(df):,} rows from {data_source}.")
    st.dataframe(df.head(10), width="stretch", hide_index=True)
