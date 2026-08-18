# Machine Learning Assignment 2

## a. Problem statement

Build a binary classification workflow using a public dataset, compare the specified machine-learning classifiers on the same test set, and present the results in a Streamlit application.

The task is breast-tumor diagnosis:

- `1 = malignant`
- `0 = benign`

## b. Dataset description

**Dataset:** Breast Cancer Wisconsin (Diagnostic)<br>
**Source:** UCI Machine Learning Repository<br>
**Dataset link:** https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

The project uses the scikit-learn packaged copy of the dataset. It contains 569 records and 30 numeric input features.

- Training set: 455 records (80%)
- Test set: 114 records (20%)
- Split: stratified
- `random_state = 42`

`test_data.csv` contains the held-out test records used for evaluation.

## c. Project links

**GitHub repository:** https://github.com/nsverma/ML_Assignment_02<br>
**Streamlit application:** https://sudhanshuverma.streamlit.app

## d. Models and evaluation

The following models were evaluated:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (kNN)
4. Gaussian Naive Bayes
5. Random Forest

### Model comparison

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.9737 | 0.9942 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| kNN | 0.9561 | 0.9825 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Decision Tree | 0.8772 | 0.9654 | 0.9118 | 0.7381 | 0.8158 | 0.7343 |

All metrics were calculated on the same held-out test set. Malignant is the positive class.

### Model observations

| Model | Observation |
|---|---|
| Logistic Regression | Produced the highest AUC and strong MCC after feature standardization. |
| Decision Tree | Had the lowest recall and MCC on the held-out test set. |
| kNN | Performed strongly after feature standardization. |
| Naive Bayes | Produced high AUC but lower recall and MCC than the strongest models. |
| Random Forest | Achieved the highest accuracy, F1, and MCC. |

Random Forest is the best overall model on this test set because it achieved the highest MCC and F1.

## Execution

Install the dependencies:

```bash
pip install -r requirements.txt
```

Train the models:

```bash
python train_models.py
```

Run the Streamlit application:

```bash
streamlit run app.py
```
