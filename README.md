# 💳 Credit Card & Transaction Fraud Detection — End-to-End ML Project

An end-to-end Machine Learning project designed to detect fraudulent financial transactions using human-interpretable features, trained with **XGBoost** and **SMOTE**, and deployed as an interactive **Streamlit** web application.

> **Disclaimer:** This is an academic and portfolio Machine Learning demonstration project. It is not a live banking system.

---

## 📌 Project Overview

Traditional credit card fraud detection benchmarks (such as the ULB dataset) obscure original transaction fields via Principal Component Analysis (PCA), providing only abstract components ($V1$ through $V28$). While valuable for mathematical benchmarking, those anonymized features cannot be entered or interpreted by real end users.

This project addresses that gap by building a **fully transparent, user-facing fraud detection system** based on the **PaySim** mobile money transaction schema. Users can upload standard transaction CSV files or input single transactions with meaningful fields (transaction type, amount, sender/recipient balance changes) and receive real-time fraud probability scores and classifications.

---

## 🎯 Problem Statement

Financial fraud detection represents an extreme class imbalance problem: typically, fewer than **0.15% to 0.20%** of all transactions are fraudulent. 

A naive classifier that labels every transaction as "genuine" achieves **99.85%+ accuracy** while missing **100% of fraud attacks**. Consequently:
- **Accuracy is an uninformative metric.**
- Models must be evaluated on **Precision**, **Recall**, **F1-Score**, and **Precision-Recall AUC (PR-AUC)**.
- The decision threshold must be calibrated specifically for class imbalance rather than defaulting to 0.50.

---

## 📊 Dataset Schema

The user-facing pipeline uses realistic financial attributes:

| Feature | Type | Description |
|---------|------|-------------|
| `type` | Categorical | Transaction category: `CASH-OUT`, `PAYMENT`, `CASH-IN`, `TRANSFER`, `DEBIT` |
| `amount` | Numeric | Transaction value in currency units ($) |
| `oldbalanceOrg` | Numeric | Initial account balance of sender before the transaction |
| `newbalanceOrig` | Numeric | Final account balance of sender after the transaction |
| `oldbalanceDest` | Numeric | Initial account balance of recipient before the transaction |
| `newbalanceDest` | Numeric | Final account balance of recipient after the transaction |
| `transaction_id` | String | *(Optional)* Unique transaction tracking identifier (e.g. `TX001`) |

---

## ⚙️ Data Preprocessing & Feature Engineering

Before passing transaction data to the classifier, domain-specific feature engineering captures known behavioral indicators of financial fraud:

1. **`balance_diff_orig`**: Discrepancy between sender balance drop and transaction amount ($\text{newbalanceOrig} - \text{oldbalanceOrg} + \text{amount}$).
2. **`balance_diff_dest`**: Discrepancy between recipient balance gain and transaction amount ($\text{newbalanceDest} - \text{oldbalanceDest} - \text{amount}$).
3. **`orig_drained`**: Binary indicator flagging whether the sender account was completely emptied ($\text{newbalanceOrig} = 0$).
4. **`dest_unchanged`**: Binary indicator flagging whether the recipient balance remained unchanged despite incoming funds (common in fraudulent mule/shell accounts).
5. **`amount_to_balance`**: Ratio of transaction amount relative to sender available balance ($\frac{\text{amount}}{\text{oldbalanceOrg} + 1}$).

Categorical variables (`type`) are encoded via `OneHotEncoder(handle_unknown='ignore')`, while numerical variables are scaled using `StandardScaler`. All transformations are bundled into a Scikit-Learn `ColumnTransformer`.

---

## ⚖️ Handling Class Imbalance

To prevent model bias toward the genuine majority class:
- The data is split into **Train (60%)**, **Validation (20%)**, and **Test (20%)** sets using stratified sampling.
- **SMOTE (Synthetic Minority Over-sampling Technique)** is applied **strictly to the training fold only**.
- Validation and test splits remain completely untouched to ensure zero data leakage.

---

## 🤖 Model & Threshold Selection

An **XGBoost Classifier** (`XGBClassifier`) is trained on the resampled training data:
- `n_estimators`: 400
- `max_depth`: 7
- `learning_rate`: 0.05
- `subsample`: 0.80
- `colsample_bytree`: 0.80
- `min_child_weight`: 5
- `eval_metric`: `aucpr`

### Optimal Threshold Tuning
Rather than using an arbitrary 0.50 cutoff, the decision threshold is tuned over the validation set to maximize the F1-Score:
- **Calibrated Threshold:** $\approx \mathbf{0.25}$
- Transactions with estimated fraud probability $\ge 0.25$ are classified as **FRAUD**; otherwise **GENUINE**.

---

## 📈 Evaluation Metrics (Untouched Test Set)

Evaluated on **40,000 completely untouched test transactions**:

| Metric | Score | Description |
|--------|-------|-------------|
| **Accuracy** | **99.990%** | Overall correct rate |
| **Precision** | **100.00%** | Flagged transactions that were genuinely fraud |
| **Recall** | **93.33%** | Actual fraud transactions successfully detected |
| **F1-Score** | **96.55%** | Harmonic mean of precision and recall |
| **ROC-AUC** | **99.99%** | Area under the ROC curve |
| **PR-AUC** | **96.63%** | Precision-Recall curve area (critical for imbalanced fraud) |

### Test Confusion Matrix
```
                  Predicted Genuine   Predicted Fraud
Actual Genuine         39,940                0
Actual Fraud                4               56
```
- **True Negatives (TN):** 39,940
- **False Positives (FP):** 0
- **False Negatives (FN):** 4
- **True Positives (TP):** 56

---

## 💻 Streamlit Web Application

The application provides two complementary inference interfaces:

1. **📁 Batch CSV Upload (Primary Mode):**
   - Upload any CSV containing transaction records.
   - Built-in schema validation (flags missing columns, invalid transaction types, negative amounts, or malformed data).
   - Real-time scoring across all rows.
   - Interactive data table highlighting fraudulent transactions in red.
   - **Download Predictions CSV** with appended `Fraud Probability (%)` and `Prediction` columns.
   - **Download Sample CSV** button for immediate testing.

2. **🔍 Single Transaction Entry:**
   - Enter individual transaction details (type, amount, sender/recipient balance before and after).
   - Generates an immediate visual verdict (**🚨 FRAUD** or **✅ GENUINE**), exact probability percentage, and interactive Plotly gauge.

3. **📊 Model Performance Page:**
   - Interactive confusion matrix heatmap and metric cards detailing untouched test set benchmarks.

4. **ℹ️ About Page:**
   - Comprehensive background on the ML architecture, feature definitions, and project limitations.

---

## 📋 Input CSV Format

Uploaded files must contain the following columns (order does not matter):

```csv
transaction_id,type,amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest
TX001,PAYMENT,1200.50,15000.00,13799.50,0.00,1200.50
TX002,CASH-IN,5000.00,2000.00,2000.00,8000.00,13000.00
TX003,TRANSFER,980000.00,980500.00,0.00,1200.00,1200.00
```

- `transaction_id`: Optional identifier.
- `type`: Must be one of `CASH-OUT`, `CASH-IN`, `PAYMENT`, `TRANSFER`, `DEBIT`.
- All monetary fields must be non-negative numbers.

---

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Aridaman1912/CreditCardFraudDetection.git
   cd CreditCardFraudDetection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

4. **Run the test suite:**
   ```bash
   python test_suite.py
   ```

---

## 📁 Repository Structure

```
CreditCardFraudDetection/
│
├── app.py                                   # Streamlit web application
├── requirements.txt                         # Application dependencies
├── README.md                                # Project documentation
├── test_suite.py                            # Automated test suite (11 unit tests)
│
├── models/
│   ├── paysim_pipeline.pkl                  # Complete user-facing preprocessor + model artifact
│   └── fraud_detection_model.pkl            # Original research V1-V28 model artifact
│
├── src/
│   ├── __init__.py
│   └── predict.py                           # Inference, feature extraction, and validation logic
│
├── data/
│   ├── sample_user_transactions.csv         # Ready-to-upload demo transactions CSV
│   └── sample_transactions.csv              # Research dataset samples
│
└── notebooks/
    ├── User_Facing_Fraud_Model.ipynb        # User-facing PaySim model training notebook
    ├── train_paysim_model.py                # Standalone training pipeline script
    └── Credit_Card_Fraud_Detection.ipynb    # Original research exploratory notebook
```

---

## ⚠️ Limitations & Ethical Considerations

- **Anonymized Features:** The historical European cardholder dataset ($V1$–$V28$) cannot be reversibly mapped to plain English merchant/location attributes. Attempting to fake such a conversion would be scientifically unsound.
- **PaySim Schema:** This user-facing system operates on transactional flow and balance dynamics, which are standard in fraud engineering. In production enterprise banking, fraud engines combine transactional flow with device telemetry, IP geolocation, biometrics, and cardholder history.
