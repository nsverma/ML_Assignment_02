"""Train and save all assignment models on one classification dataset.

Dataset: Breast Cancer Wisconsin (Diagnostic), UCI. The scikit-learn packaged
copy is used for reproducibility in BITS Virtual Lab and Streamlit Cloud.
Target is remapped so 1=malignant and 0=benign.
"""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
RESULTS_DIR = ROOT / "results"
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def prepare_data():
    bunch = load_breast_cancer(as_frame=True)
    df = bunch.frame.copy()
    # scikit-learn target: 0 malignant, 1 benign.
    # Remap so the positive class (1) means malignant.
    df["diagnosis"] = (df["target"] == 0).astype(int)
    df = df.drop(columns=["target"])
    features = [c for c in df.columns if c != "diagnosis"]
    X_train, X_test, y_train, y_test = train_test_split(
        df[features],
        df["diagnosis"],
        test_size=0.20,
        random_state=42,
        stratify=df["diagnosis"],
    )
    test_df = X_test.copy()
    test_df["diagnosis"] = y_test.values
    test_df.to_csv(ROOT / "test_data.csv", index=False)
    df.to_csv(ROOT / "full_dataset.csv", index=False)
    return df, features, X_train, X_test, y_train, y_test


def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, random_state=42)),
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=4, random_state=42
        ),
        "kNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=7)),
        ]),
        "Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GaussianNB()),
        ]),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=400,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
    }


MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest.joblib",
}


def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": accuracy_score(y_test, pred),
        "AUC": roc_auc_score(y_test, prob),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, pred),
    }


def main():
    df, features, X_train, X_test, y_train, y_test = prepare_data()
    models = build_models()
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_DIR / MODEL_FILES[name])
        row = {"ML Model Name": name, **evaluate(model, X_test, y_test)}
        rows.append(row)
        print(name, row)

    metrics = pd.DataFrame(rows).sort_values(
        ["MCC", "F1", "AUC"], ascending=False
    )
    metrics.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)

    metadata = {
        "dataset_name": "Breast Cancer Wisconsin (Diagnostic)",
        "dataset_source": "UCI Machine Learning Repository; scikit-learn packaged copy",
        "dataset_url": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
        "instances": int(df.shape[0]),
        "features": len(features),
        "target": "diagnosis",
        "target_mapping": {"0": "benign", "1": "malignant"},
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "random_state": 42,
        "feature_columns": features,
        "model_files": MODEL_FILES,
        "sklearn_version": sklearn_version,
    }
    (MODEL_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("\nSaved models, metrics, full_dataset.csv and test_data.csv.")


if __name__ == "__main__":
    main()
