# 🛡️ Credit Card Fraud Detection - ML Prediction Web App

An end-to-end Machine Learning web application built using **Streamlit** and **XGBoost** for real-time transaction risk prediction and fraud classification.

---

## 📌 Project Overview

Credit card fraud is characterized by an extreme class imbalance, where fraudulent transactions represent a tiny fraction of total activity. This project takes the final trained research model and deploys it into an interactive prediction web application where users can:
- Input transaction parameters to immediately compute fraud probability.
- Classify transactions as **FRAUD** or **GENUINE** using an optimized decision threshold.
- Perform batch scoring on uploaded CSV datasets.
- Review model performance metrics, confusion matrix, and SHAP explainability insights.

---

## 📊 Dataset Details

- **Dataset**: `creditcard.csv`
- **Total Records**: 284,807 transactions (European cardholders over two days)
- **Features (30 inputs)**:
  - `Time`: Elapsed seconds from the initial recorded transaction.
  - `V1` – `V28`: Anonymized numerical features resulting from Principal Component Analysis (PCA) to protect cardholder privacy.
  - `Amount`: Transaction transaction amount in USD.
- **Target (`Class`)**:
  - `0`: Genuine / Legitimate
  - `1`: Fraud (only 492 cases, representing **0.172%** of all transactions)

---

## 🧠 Machine Learning Approach

### 1. The Class Imbalance Problem
Because 99.828% of transactions are genuine, standard models suffer from severe majority-class bias. A trivial baseline predicting all transactions as genuine would achieve 99.83% accuracy while failing to intercept any fraud.
To address this:
- **SMOTE (Synthetic Minority Over-sampling Technique)** was applied strictly on the training fold to generate synthetic fraud instances along minority class feature vectors.
- Features were standardized using `StandardScaler`.

### 2. Tuned XGBoost Model
The final model is an optimized **XGBoost Classifier** with the following key hyperparameters:
- `n_estimators`: 300
- `max_depth`: 11
- `learning_rate`: 0.10
- `subsample`: 0.80
- `colsample_bytree`: 0.80
- `min_child_weight`: 5
- `eval_metric`: logloss

### 3. Threshold Optimization
Default binary classification uses a threshold of $0.50$. For extreme class imbalance and financial risk trade-offs:
- The decision boundary was tuned to **`0.89`**.
- This eliminates more than 80% of false alarms (False Positives) while maintaining high fraud recall, minimizing both direct financial losses and customer friction.

---

## 📈 Final Untouched Test-Set Performance

Evaluated on 56,962 untouched test transactions using threshold **`0.89`**:

| Metric | Score |
| :--- | :---: |
| **Accuracy** | **99.956%** |
| **Precision** | **91.95%** |
| **Recall** | **81.63%** |
| **F1 Score** | **86.49%** |
| **ROC-AUC** | **98.11%** |
| **PR-AUC** | **87.51%** |

### Confusion Matrix
```
                     Predicted Genuine (0)    Predicted Fraud (1)
Actual Genuine (0)           56,857                    7
Actual Fraud (1)               18                     80
```
- **True Negatives (TN)**: 56,857
- **False Positives (FP)**: 7
- **False Negatives (FN)**: 18
- **True Positives (TP)**: 80

---

## 📁 Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── data/
│   └── creditcard.csv                       # Transaction dataset (284,807 rows)
│
├── models/
│   └── fraud_detection_model.pkl            # Final model artifact (model, scaler, feature_names, threshold)
│
├── notebooks/
│   └── Credit_Card_Fraud_Detection.ipynb    # Complete research, tuning & evaluation notebook
│
├── src/
│   └── predict.py                           # Dedicated prediction, validation & batch scoring logic
│
├── app.py                                   # Streamlit prediction web application
├── requirements.txt                         # Application dependencies
└── README.md                                # Project documentation
```

---

## 🚀 How to Run the Streamlit Application

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Application
```bash
streamlit run app.py
```

Open your browser at **`http://localhost:8501`**.

---

## 🖥️ Application Features & User Flow

1. **Transaction Risk Prediction (Primary Feature)**:
   - Enter `Time`, `Amount`, and PCA features `V1` to `V28` (or use the 1-click **Load Genuine Sample** / **Load Fraud Sample** presets).
   - Click **`🔍 Predict Transaction`**.
   - Displays:
     - **`Fraud Probability: XX.XX%`**
     - **`⚠️ FRAUD DETECTED`** (High Risk, prob $\ge 0.89$) or **`✅ GENUINE TRANSACTION`** (Low Risk, prob $< 0.89$).
     - Visual risk gauge indicator.
2. **Model Performance**:
   - Displays cards for Accuracy (99.956%), Precision (91.95%), Recall (81.63%), F1 (86.49%), ROC-AUC (98.11%), and PR-AUC (87.51%).
3. **Confusion Matrix**:
   - Interactive heatmap displaying the test set confusion matrix `[[56857, 7], [18, 80]]`.
4. **Batch Prediction**:
   - Upload any CSV containing the 30 features (or test with the built-in 200-transaction demo batch).
   - Instant automated classification and CSV export.
5. **Model Information & Explainability**:
   - Specifications table and top 10 SHAP high-impact features (`V14`, `V4`, `V8`, `V12`, `V10`, `V1`, `V18`, `V3`, `V11`, `V22`).
