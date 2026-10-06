# Credit Card Fraud Detection — ML Portfolio Project

An end-to-end machine learning project that trains an **XGBoost** classifier on the public
ULB Credit Card Fraud dataset and deploys it as an interactive **Streamlit** web application.

> **This is an ML portfolio / learning project.** It is not a real banking production system.

---

## Project Objective

Detect fraudulent credit card transactions using machine learning while addressing the severe
class imbalance (only 0.173% of transactions are fraudulent) that makes standard accuracy metrics
misleading.

---

## Dataset

| Property | Value |
|----------|-------|
| Source | ULB Machine Learning Group (Kaggle) |
| Period | September 2013, European cardholders |
| Transactions | 284,807 over 48 hours |
| Fraud cases | 492 (0.173%) |
| Features | `Time`, `V1`–`V28`, `Amount`, `Class` |

**Important — V1 to V28 are anonymized features.**
They are the result of a PCA transformation applied by the original dataset authors to protect
cardholder privacy. Their individual real-world meanings are not disclosed or interpretable.

---

## Class Imbalance Problem

Because only 0.173% of transactions are fraudulent, a naive model that always predicts "genuine"
achieves 99.83% accuracy while catching **zero frauds**. To address this:

- **SMOTE** (Synthetic Minority Over-sampling Technique) is applied to the training fold only.
- Evaluation focuses on **Precision**, **Recall**, **F1**, and **PR-AUC** rather than accuracy.
- The decision threshold is tuned away from the default 0.50.

---

## Machine Learning Workflow

1. Stratified train / test split (80 / 20)
2. `StandardScaler` fitted on training data only — applied to all splits
3. SMOTE applied to the scaled training fold to balance classes
4. XGBoost trained on the resampled training set
5. Decision threshold tuned on validation data to maximise F1
6. Final evaluation on the untouched test set

### XGBoost Hyperparameters (tuned)

| Parameter | Value |
|-----------|-------|
| n_estimators | 300 |
| max_depth | 11 |
| learning_rate | 0.10 |
| subsample | 0.80 |
| colsample_bytree | 0.80 |
| min_child_weight | 5 |
| eval_metric | logloss |

---

## Final Test-Set Performance

> Evaluated on 56,962 untouched test transactions at threshold **0.89**.

| Metric | Score |
|--------|-------|
| Accuracy | **99.956%** |
| Precision | **91.95%** |
| Recall | **81.63%** |
| F1 Score | **86.49%** |
| ROC-AUC | **98.11%** |
| PR-AUC | **87.51%** |

### Confusion Matrix

```
                  Predicted Genuine   Predicted Fraud
Actual Genuine         56,857               7
Actual Fraud               18              80
```

- **True Negatives:** 56,857
- **False Positives:** 7
- **False Negatives:** 18
- **True Positives:** 80

---

## Threshold Optimization

The default classification threshold is 0.50. Because the cost of missing real fraud (False Negative)
far outweighs the cost of a false alarm (False Positive), the threshold is calibrated to **0.89**
to maximise the F1 score on the validation set.

---

## Streamlit Application

The web application provides:

| Page | Description |
|------|-------------|
| **🔍 Prediction** | Try sample transactions or upload a CSV — no manual V1–V28 entry required |
| **📁 Batch Prediction** | Score hundreds of transactions at once, download results |
| **📊 Model Performance** | Metrics, confusion matrix, dataset imbalance explanation |
| **🧬 Explainability** | SHAP feature importance rankings |
| **ℹ️ About** | Project background and instructions |

### Primary User Flow

```
Try Sample Transaction  OR  Upload CSV
         ↓
   Validate columns
         ↓
  Reorder by feature_names
         ↓
  Saved StandardScaler
         ↓
  Saved XGBoost model
         ↓
     predict_proba()
         ↓
  Threshold = 0.89
         ↓
  FRAUD / GENUINE result
```

---

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at **http://localhost:8501**.

---

## How to Upload Your Own CSV

The CSV must contain these 30 columns (order does not matter, the app reorders them):

```
Time, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10,
V11, V12, V13, V14, V15, V16, V17, V18, V19, V20,
V21, V22, V23, V24, V25, V26, V27, V28, Amount
```

- A `Class` column is **ignored** if present (not used as model input).
- Extra columns are ignored.
- Missing required columns produce a clear error message.
- Non-numeric values produce a clear error message.

### Example CSV format

```csv
Time,V1,V2,V3,...,V28,Amount
0,-1.3598,0.0,2.536,...,-0.021,149.62
1,1.1918,0.266,0.166,...,0.014,2.69
```

---

## Project Structure

```
CreditCardFraudDetection/
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Dependencies
├── README.md                       # This file
│
├── models/
│   └── fraud_detection_model.pkl   # Trained model artifact
│                                   # (contains: model, scaler, feature_names, threshold)
│
├── src/
│   ├── __init__.py
│   └── predict.py                  # predict_fraud(), predict_dataframe(), validate_features()
│
├── data/
│   └── sample_transactions.csv     # 10 real sample rows (5 genuine, 5 fraud) for demo
│
├── sample_data/
│   ├── fraud_samples.csv           # 15 confirmed fraud rows
│   ├── legitimate_samples.csv      # 15 genuine rows
│   └── batch_test_sample.csv       # 200 mixed rows for batch demo
│
└── notebooks/
    └── Credit_Card_Fraud_Detection.ipynb   # Full research, EDA, and training notebook
```

---

## Notes

- The full `creditcard.csv` (144 MB) is not included in the repository.
- The model artifact (`fraud_detection_model.pkl`, ~1.3 MB) contains the trained XGBoost model,
  the fitted StandardScaler, the ordered feature name list, and the calibrated threshold.
- SMOTE is applied only during training — it is not re-applied at inference time.
- The scaler is not re-fitted at inference time.
