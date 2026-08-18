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
    page_title="Breast cancer model studio",
    page_icon=":material/health_metrics:",
    layout="wide",
)

# Streamlit's interactive dataframe renders its own header grid and does not
# apply Pandas Styler alignment to column headings.
st.markdown(
    """
    <style>
    div[data-testid="stDataFrame"] div[role="columnheader"] {
        justify-content: center !important;
        text-align: center !important;
    }
    div[data-testid="stDataFrame"] div[role="columnheader"] > div {
        justify-content: center !important;
        text-align: center !important;
        width: 100% !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_SUMMARIES = {
    "Logistic Regression": "Scaled linear baseline with directly interpretable malignancy probabilities.",
    "Decision Tree": "Compact nonlinear decision rules constrained to reduce overfitting.",
    "kNN": "Scaled distance-based classifier using the seven nearest training cases.",
    "Naive Bayes": "Fast probabilistic baseline with a Gaussian feature assumption.",
    "Random Forest (Ensemble)": "Four hundred decision trees combined to improve stability and generalization.",
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


def centered_headers(frame):
    """Return a table style with centered column headings."""
    return frame.style.set_table_styles(
        [{"selector": "th", "props": [("text-align", "center")]}]
    )


metadata, models = load_assets()
features = metadata["feature_columns"]

st.title("Breast cancer model studio")
st.caption(
    "Compare five classifiers on the Breast Cancer Wisconsin (Diagnostic) dataset. "
    "Malignant is treated as the positive class."
)
st.markdown(":blue-badge[30 numeric features] :violet-badge[5 classifiers] :green-badge[Stratified holdout]")

with st.sidebar:
    st.header("Evaluation controls")
    uploaded = st.file_uploader(
        "Test dataset",
        type=["csv"],
        help="Upload a CSV containing all 30 feature columns and the diagnosis target.",
    )
    selected_model = st.selectbox("Model to inspect", list(models.keys()))
    st.caption(MODEL_SUMMARIES[selected_model])
    with st.expander("Data contract", icon=":material/table_chart:"):
        st.write("`diagnosis = 1` → malignant")
        st.write("`diagnosis = 0` → benign")
        st.caption("Training split: 80/20 · stratified · random state 42")
    st.caption("Educational demonstration — not a clinical diagnostic tool.")

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
    st.error("Missing required columns: " + ", ".join(missing), icon=":material/error:")
    st.stop()

try:
    X = df[features].apply(pd.to_numeric, errors="raise")
    y_numeric = pd.to_numeric(df[metadata["target"]], errors="raise")
except (TypeError, ValueError) as exc:
    st.error(f"Features and diagnosis must be numeric: {exc}", icon=":material/error:")
    st.stop()

if set(y_numeric.unique()) != {0, 1}:
    st.error("Evaluation requires both diagnosis classes, encoded exactly as 0 and 1.", icon=":material/error:")
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
    comparison_rows.append({"Model": name, **m, "False negatives": int(model_cm[1, 0])})

comparison = pd.DataFrame(comparison_rows)
comparison = comparison.sort_values(["MCC", "F1", "AUC"], ascending=False).reset_index(drop=True)
comparison.insert(0, "Rank", comparison.index + 1)
winner = comparison.iloc[0]
counts = y.value_counts().to_dict()

summary_cols = st.columns(4)
summary_cols[0].metric("Cases", f"{len(df):,}")
summary_cols[1].metric("Malignant", int(counts.get(1, 0)))
summary_cols[2].metric("Benign", int(counts.get(0, 0)))
summary_cols[3].metric("Best MCC", f"{winner['MCC']:.4f}")
st.caption(f"Source: {data_source} · {metadata['dataset_name']}")

view = st.segmented_control(
    "Workspace",
    ["Model comparison", "Selected model", "Predictions"],
    default="Model comparison",
    label_visibility="collapsed",
)

metrics, predictions, probabilities, cm = cache[selected_model]

if view == "Model comparison":
    st.header("Model comparison")
    st.caption("Ranked by MCC, then F1 and AUC. False negatives are malignant cases predicted as benign.")

    chart_col, safety_col = st.columns([1.6, 1])
    with chart_col:
        st.subheader("Performance by model")
        chart_metrics = st.multiselect(
            "Metrics to compare",
            ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
            default=["AUC", "Recall", "F1", "MCC"],
        )
        if chart_metrics:
            performance_chart = comparison.set_index("Model")[chart_metrics]
            st.bar_chart(performance_chart, y=chart_metrics, height=380)
        else:
            st.info("Select at least one metric to display the comparison chart.")

    with safety_col:
        st.subheader("Missed malignant cases")
        safety_chart = comparison.set_index("Model")[["False negatives"]]
        st.bar_chart(safety_chart, y="False negatives", height=380)
        st.caption("Lower is better. A false negative is a malignant case predicted as benign.")

    st.subheader("Detailed results")
    comparison_table = centered_headers(comparison).format(
        {c: "{:.4f}" for c in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]}
    )
    st.dataframe(
        comparison_table,
        width="stretch",
        hide_index=True,
    )
    st.success(
        f"{winner['Model']} leads this test set with MCC {winner['MCC']:.4f}.",
        icon=":material/verified:",
    )
    with st.expander("Preview evaluation data", icon=":material/database:"):
        st.dataframe(centered_headers(df.head(10)), width="stretch", hide_index=True)

elif view == "Selected model":
    st.header(selected_model)
    st.caption(MODEL_SUMMARIES[selected_model])
    top_metrics = st.columns(3)
    top_metrics[0].metric("Malignant recall", f"{metrics['Recall']:.4f}")
    top_metrics[1].metric("Malignant precision", f"{metrics['Precision']:.4f}")
    top_metrics[2].metric("MCC", f"{metrics['MCC']:.4f}")
    secondary_metrics = st.columns(3)
    secondary_metrics[0].metric("Accuracy", f"{metrics['Accuracy']:.4f}")
    secondary_metrics[1].metric("AUC", f"{metrics['AUC']:.4f}")
    secondary_metrics[2].metric("F1", f"{metrics['F1']:.4f}")

    tn, fp, fn, tp = cm.ravel()
    matrix_col, report_col = st.columns([0.9, 1.4])
    with matrix_col:
        with st.container(border=True):
            st.subheader("Confusion matrix")
            cm_df = pd.DataFrame(
                cm,
                index=["Actual benign", "Actual malignant"],
                columns=["Predicted benign", "Predicted malignant"],
            )
            st.dataframe(centered_headers(cm_df), width="stretch")
            outcome_cols = st.columns(2)
            outcome_cols[0].metric("Missed malignant", int(fn), help="False negatives")
            outcome_cols[1].metric("Benign flagged", int(fp), help="False positives")
    with report_col:
        st.subheader("Classification report")
        report = classification_report(
            y,
            predictions,
            labels=[0, 1],
            target_names=["Benign", "Malignant"],
            output_dict=True,
            zero_division=0,
        )
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(centered_headers(report_df).format("{:.4f}"), width="stretch")

    if fn:
        st.warning(f"This model missed {int(fn)} malignant case(s) in the current test data.", icon=":material/warning:")
    else:
        st.success("No malignant cases were missed in the current test data.", icon=":material/check_circle:")

else:
    st.header("Prediction review")
    preview = pd.DataFrame({
        "Actual": y.map({0: "Benign", 1: "Malignant"}).values,
        "Predicted": pd.Series(predictions).map({0: "Benign", 1: "Malignant"}),
        "Malignancy probability": probabilities,
    })
    preview["Outcome"] = preview.apply(
        lambda row: "Correct" if row["Actual"] == row["Predicted"] else "Error",
        axis=1,
    )
    only_errors = st.toggle("Show prediction errors only")
    visible_preview = preview[preview["Outcome"] == "Error"] if only_errors else preview
    st.dataframe(
        centered_headers(visible_preview),
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
