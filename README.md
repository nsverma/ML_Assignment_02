# Machine Learning Assignment 2 - Diagnostic Classification Lab

## a. Problem statement

The goal is to build an end-to-end binary classification workflow on one public dataset, compare the assignment-specified machine-learning classifiers using the same test split, expose the evaluation through an interactive Streamlit application, and deploy the app on Streamlit Community Cloud.

The prediction task used here is **breast-tumor diagnosis**. The target is intentionally remapped to make the positive class medically intuitive:

- `1 = malignant`
- `0 = benign`

## b. Dataset description

**Dataset:** Breast Cancer Wisconsin (Diagnostic)  
**Original public source:** UCI Machine Learning Repository  
**Dataset link:** https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

The project uses the scikit-learn packaged copy (`load_breast_cancer`) for reproducible execution. It contains **569 instances** and **30 numeric input features**, satisfying the assignment minima of at least 500 instances and at least 12 features.

Split used:

- Training: **455 rows (80%)**
- Test: **114 rows (20%)**
- Stratified split
- `random_state = 42`

`test_data.csv` contains the exact held-out test records used to generate the reported metrics.

## c. GitHub Repository Link

**GitHub Repository:** `PASTE_YOUR_GITHUB_REPOSITORY_LINK_HERE`

## d. Models used and evaluation

> The assignment PDF says that "all 6 ML models" are required, but the model list and the comparison table enumerate **five** models. This project implements all five models explicitly named in the brief: Logistic Regression, Decision Tree, kNN, Naive Bayes, and Random Forest.

### Models

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (kNN) Classifier
4. Gaussian Naive Bayes
5. Random Forest (Ensemble)

### Comparison table

| ML Model Name            |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:-------------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Random Forest (Ensemble) |     0.9737 | 0.9942 |      1      |   0.9286 | 0.963  | 0.9442 |
| Logistic Regression      |     0.9649 | 0.996  |      0.975  |   0.9286 | 0.9512 | 0.9245 |
| kNN                      |     0.9561 | 0.9825 |      0.9744 |   0.9048 | 0.9383 | 0.9058 |
| Naive Bayes              |     0.9211 | 0.9891 |      0.9231 |   0.8571 | 0.8889 | 0.8292 |
| Decision Tree            |     0.8772 | 0.9654 |      0.9118 |   0.7381 | 0.8158 | 0.7343 |

Metrics are calculated on the same held-out test set. AUC uses the predicted probability of the positive class (`malignant = 1`).

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Very strong linear baseline after standardization. It achieved excellent AUC and a high MCC, showing that the classes are close to linearly separable in the standardized feature space. |
| Decision Tree | The single tree was the weakest of the five tested models on the held-out set. Its lower recall and MCC indicate greater sensitivity to split choices and a higher risk of overfitting or unstable boundaries. |
| kNN | kNN performed strongly after standardization. Its neighborhood-based decision rule captured the local structure well, although it remained slightly behind the best ensemble on the held-out test set. |
| Naive Bayes | Gaussian Naive Bayes produced a good AUC but lower accuracy/MCC than the strongest models. The conditional-independence assumption is restrictive because many tumor measurements are correlated. |
| Random Forest (Ensemble) | Random Forest delivered the best overall held-out performance by MCC and F1, combining many trees to reduce variance while retaining nonlinear decision boundaries. |
| **Overall Winner** | **Random Forest (Ensemble)** is the overall winner on this held-out test set because it achieved the highest MCC and F1 among the tested models while also maintaining excellent AUC. |

## Streamlit application features

The application implements all minimum UI features requested in the assignment:

- CSV test-data upload
- Model-selection dropdown
- Evaluation metrics: Accuracy, AUC, Precision, Recall, F1 and MCC
- All-model comparison table
- Confusion matrix
- Classification report
- Prediction preview
- Results re-computed on the uploaded compatible test CSV

## Repository structure

```text
ml_assignment_2_breast_cancer/
|-- app.py
|-- train_models.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- full_dataset.csv
|-- model/
|   |-- logistic_regression.joblib
|   |-- decision_tree.joblib
|   |-- knn.joblib
|   |-- naive_bayes.joblib
|   |-- random_forest.joblib
|   `-- metadata.json
|-- results/
|   `-- model_metrics.csv
`-- notebooks/
    `-- ML_Assignment_2.ipynb
```

## How to execute on BITS Virtual Lab

1. Upload or clone this repository in the BITS Virtual Lab.
2. Open a terminal in the project folder.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Re-train and reproduce the metrics:

```bash
python train_models.py
```

5. Start Streamlit:

```bash
streamlit run app.py
```

6. Open the displayed local URL and verify the app.
7. Take **one genuine screenshot from the BITS Virtual Lab** showing successful assignment execution, as required by the brief.

## Streamlit Community Cloud deployment

1. Push the project to your own GitHub repository.
2. Sign in to Streamlit Community Cloud with GitHub.
3. Create a new app from your repository.
4. Select the `main` branch.
5. Use `app.py` as the entry point.
6. Deploy and test `test_data.csv` through the upload control.

**Live Streamlit App:** `PASTE_YOUR_STREAMLIT_APP_LINK_HERE`

## Reproducibility note

The saved models were generated with scikit-learn **1.8.0**. `requirements.txt` pins the same scikit-learn version to avoid model-deserialization mismatch.

## Academic integrity / personalization note

The assignment brief states that GitHub history, variable names, UI similarity, dataset/model/output similarity, and copy-paste templates may be checked. Before submission, run the work yourself in BITS Virtual Lab, understand each section, use your own GitHub commit history, and personalize the UI wording/README observations rather than submitting unchanged AI-generated material.
