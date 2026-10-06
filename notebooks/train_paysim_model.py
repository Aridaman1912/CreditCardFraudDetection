"""
Train a fraud detection model on PaySim-schema synthetic data.
Features: type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest
Target: isFraud

Pipeline saved to: models/paysim_pipeline.pkl
"""

import pickle, warnings, os
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
np.random.seed(42)

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("notebooks", exist_ok=True)

# ─────────────────────────────────────────────────────────
# 1. GENERATE SYNTHETIC PAYSIM-SCHEMA DATA
#    Distributions calibrated to the real PaySim statistics:
#    - 6.3M rows, 0.13% fraud
#    - Fraud concentrated in CASH-OUT and TRANSFER types
#    - Fraud cases: balance anomalies (origin drained to 0)
# ─────────────────────────────────────────────────────────
print("Generating synthetic PaySim-schema dataset...")

N = 200_000          # rows for fast training — enough to learn robust patterns
FRAUD_RATE = 0.0015  # ~0.15%, close to real PaySim

rng = np.random.default_rng(42)

# Transaction types and their rough distribution
TYPES  = ["CASH-OUT", "TRANSFER", "PAYMENT", "CASH-IN", "DEBIT"]
TPROBS = [0.35,        0.08,       0.34,      0.22,      0.01]

# ── Genuine transactions ──────────────────────────────────
n_genuine = int(N * (1 - FRAUD_RATE))
n_fraud   = N - n_genuine

def gen_genuine(n):
    types = rng.choice(TYPES, size=n, p=TPROBS)
    amount = np.abs(rng.exponential(scale=180_000, size=n))
    amount = np.clip(amount, 1, 10_000_000)

    oldbalOrg  = np.abs(rng.exponential(scale=800_000, size=n))
    newbalOrg  = np.maximum(oldbalOrg - amount, 0)

    # Payment and cash-in don't reduce originator balance
    is_payment = np.isin(types, ["PAYMENT", "CASH-IN"])
    newbalOrg  = np.where(is_payment, oldbalOrg, newbalOrg)

    oldbalDest = np.abs(rng.exponential(scale=600_000, size=n))
    newbalDest = oldbalDest + amount

    return pd.DataFrame({
        "type":            types,
        "amount":          np.round(amount, 2),
        "oldbalanceOrg":   np.round(oldbalOrg, 2),
        "newbalanceOrig":  np.round(newbalOrg, 2),
        "oldbalanceDest":  np.round(oldbalDest, 2),
        "newbalanceDest":  np.round(newbalDest, 2),
        "isFraud":         0,
    })

# ── Fraudulent transactions (CASH-OUT & TRANSFER only) ───
def gen_fraud(n):
    types  = rng.choice(["CASH-OUT", "TRANSFER"], size=n, p=[0.5, 0.5])
    amount = np.abs(rng.exponential(scale=500_000, size=n))
    amount = np.clip(amount, 100, 10_000_000)

    # 80% of fraud: originator balance completely drained (strong signal)
    # 20% of fraud: partial drain (ambiguous / realistic noise)
    drain_fully = rng.random(n) < 0.80
    oldbalOrg   = amount + rng.exponential(scale=5_000, size=n)
    newbalOrg   = np.where(drain_fully, 0, oldbalOrg - amount * rng.uniform(0.5, 0.9, n))
    newbalOrg   = np.maximum(newbalOrg, 0)

    # 70% of fraud: destination balance unchanged (mule accounts)
    # 30% of fraud: destination balance updated normally (harder to detect)
    dest_unchanged = rng.random(n) < 0.70
    oldbalDest  = np.abs(rng.exponential(scale=10_000, size=n))
    newbalDest  = np.where(dest_unchanged, oldbalDest, oldbalDest + amount)

    return pd.DataFrame({
        "type":            types,
        "amount":          np.round(amount, 2),
        "oldbalanceOrg":   np.round(oldbalOrg, 2),
        "newbalanceOrig":  np.round(newbalOrg, 2),
        "oldbalanceDest":  np.round(oldbalDest, 2),
        "newbalanceDest":  np.round(newbalDest, 2),
        "isFraud":         1,
    })

df_genuine = gen_genuine(n_genuine)
df_fraud   = gen_fraud(n_fraud)
df = pd.concat([df_genuine, df_fraud], ignore_index=True).sample(frac=1, random_state=42)

print(f"  Total rows:   {len(df):,}")
print(f"  Genuine rows: {(df['isFraud']==0).sum():,}")
print(f"  Fraud rows:   {(df['isFraud']==1).sum():,}")
print(f"  Fraud rate:   {df['isFraud'].mean()*100:.3f}%")

# ─────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────
print("\nAdding engineered features...")

def add_features(df):
    df = df.copy()
    df["balance_diff_orig"] = df["newbalanceOrig"] - df["oldbalanceOrg"] + df["amount"]
    df["balance_diff_dest"] = df["newbalanceDest"] - df["oldbalanceDest"] - df["amount"]
    df["orig_drained"]      = (df["newbalanceOrig"] == 0).astype(int)
    df["dest_unchanged"]    = (df["newbalanceDest"] == df["oldbalanceDest"]).astype(int)
    df["amount_to_balance"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    return df

df = add_features(df)

FEATURE_COLS = [
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "balance_diff_orig",
    "balance_diff_dest",
    "orig_drained",
    "dest_unchanged",
    "amount_to_balance",
]
TARGET = "isFraud"

X = df[FEATURE_COLS]
y = df[TARGET]

# ─────────────────────────────────────────────────────────
# 3. TRAIN / VALIDATION / TEST SPLIT
# ─────────────────────────────────────────────────────────
print("Splitting data (60/20/20)...")
X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.25, stratify=y_tv, random_state=42)
print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

# ─────────────────────────────────────────────────────────
# 4. PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────
cat_cols = ["type"]
num_cols = [c for c in FEATURE_COLS if c not in cat_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ],
    remainder="drop",
)

# ─────────────────────────────────────────────────────────
# 5. APPLY SMOTE ON TRAINING DATA ONLY
# ─────────────────────────────────────────────────────────
print("Applying SMOTE on training data...")
X_train_pre = preprocessor.fit_transform(X_train)
X_val_pre   = preprocessor.transform(X_val)
X_test_pre  = preprocessor.transform(X_test)

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_res, y_train_res = smote.fit_resample(X_train_pre, y_train)
print(f"  After SMOTE — fraud: {y_train_res.sum():,}  genuine: {(y_train_res==0).sum():,}")

# ─────────────────────────────────────────────────────────
# 6. TRAIN XGBOOST
# ─────────────────────────────────────────────────────────
print("\nTraining XGBoost...")
model = XGBClassifier(
    n_estimators=400,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    eval_metric="aucpr",
    random_state=42,
    n_jobs=-1,
)
model.fit(
    X_train_res, y_train_res,
    eval_set=[(X_val_pre, y_val)],
    verbose=False,
)
print("  Training complete.")

# ─────────────────────────────────────────────────────────
# 7. THRESHOLD SELECTION ON VALIDATION SET
# ─────────────────────────────────────────────────────────
print("\nSelecting threshold on validation set...")
val_proba = model.predict_proba(X_val_pre)[:, 1]
best_f1, best_thr = 0, 0.5
for thr in np.arange(0.05, 0.95, 0.01):
    preds = (val_proba >= thr).astype(int)
    f1 = f1_score(y_val, preds, zero_division=0)
    if f1 > best_f1:
        best_f1, best_thr = f1, thr
print(f"  Best threshold: {best_thr:.2f}  (Val F1 = {best_f1:.4f})")

# ─────────────────────────────────────────────────────────
# 8. EVALUATE ON UNTOUCHED TEST SET
# ─────────────────────────────────────────────────────────
print("\nEvaluating on untouched test set...")
test_proba = model.predict_proba(X_test_pre)[:, 1]
test_preds = (test_proba >= best_thr).astype(int)

acc  = accuracy_score(y_test, test_preds)
prec = precision_score(y_test, test_preds, zero_division=0)
rec  = recall_score(y_test, test_preds, zero_division=0)
f1   = f1_score(y_test, test_preds, zero_division=0)
roc  = roc_auc_score(y_test, test_proba)
prauc = average_precision_score(y_test, test_proba)
cm   = confusion_matrix(y_test, test_preds)

print(f"  Accuracy:  {acc*100:.3f}%")
print(f"  Precision: {prec*100:.2f}%")
print(f"  Recall:    {rec*100:.2f}%")
print(f"  F1:        {f1*100:.2f}%")
print(f"  ROC-AUC:   {roc*100:.2f}%")
print(f"  PR-AUC:    {prauc*100:.2f}%")
print(f"  Confusion Matrix:\n{cm}")

# ─────────────────────────────────────────────────────────
# 9. SAVE COMPLETE PIPELINE
# ─────────────────────────────────────────────────────────
print("\nSaving pipeline...")
artifact = {
    "preprocessor":     preprocessor,
    "model":            model,
    "threshold":        float(best_thr),
    "feature_cols":     FEATURE_COLS,
    "cat_cols":         cat_cols,
    "num_cols":         num_cols,
    "user_input_cols":  ["type", "amount", "oldbalanceOrg", "newbalanceOrig",
                         "oldbalanceDest", "newbalanceDest"],
    "metrics": {
        "accuracy":  round(acc  * 100, 3),
        "precision": round(prec * 100, 2),
        "recall":    round(rec  * 100, 2),
        "f1":        round(f1   * 100, 2),
        "roc_auc":   round(roc  * 100, 2),
        "pr_auc":    round(prauc * 100, 2),
        "cm":        cm.tolist(),
        "test_size": len(X_test),
    },
}

with open("models/paysim_pipeline.pkl", "wb") as f:
    pickle.dump(artifact, f)
print("  Saved to models/paysim_pipeline.pkl")

# ─────────────────────────────────────────────────────────
# 10. CREATE SAMPLE USER TRANSACTIONS CSV
# ─────────────────────────────────────────────────────────
print("\nCreating sample_user_transactions.csv...")

sample_rows = [
    # Genuine transactions
    {"transaction_id": "TX001", "type": "PAYMENT",  "amount": 1200.50, "oldbalanceOrg": 15000, "newbalanceOrig": 13799.50, "oldbalanceDest": 0, "newbalanceDest": 1200.50},
    {"transaction_id": "TX002", "type": "CASH-IN",  "amount": 5000.00, "oldbalanceOrg": 2000,  "newbalanceOrig": 2000,     "oldbalanceDest": 8000, "newbalanceDest": 13000},
    {"transaction_id": "TX003", "type": "DEBIT",    "amount": 350.00,  "oldbalanceOrg": 9500,  "newbalanceOrig": 9150,     "oldbalanceDest": 0, "newbalanceDest": 0},
    {"transaction_id": "TX004", "type": "CASH-OUT", "amount": 2000.00, "oldbalanceOrg": 12000, "newbalanceOrig": 10000,    "oldbalanceDest": 5000, "newbalanceDest": 7000},
    {"transaction_id": "TX005", "type": "PAYMENT",  "amount": 89.99,   "oldbalanceOrg": 500,   "newbalanceOrig": 410.01,   "oldbalanceDest": 200, "newbalanceDest": 289.99},
    # Fraudulent transactions (balance-drain pattern)
    {"transaction_id": "TX006", "type": "TRANSFER", "amount": 980000,  "oldbalanceOrg": 980500, "newbalanceOrig": 0,       "oldbalanceDest": 1200, "newbalanceDest": 1200},
    {"transaction_id": "TX007", "type": "CASH-OUT", "amount": 340000,  "oldbalanceOrg": 340100, "newbalanceOrig": 0,       "oldbalanceDest": 500,  "newbalanceDest": 500},
    {"transaction_id": "TX008", "type": "TRANSFER", "amount": 75000,   "oldbalanceOrg": 75200,  "newbalanceOrig": 0,       "oldbalanceDest": 100,  "newbalanceDest": 100},
]

sample_df = pd.DataFrame(sample_rows)
sample_df.to_csv("data/sample_user_transactions.csv", index=False)
print(f"  Saved data/sample_user_transactions.csv ({len(sample_df)} rows)")

# Quick sanity check on sample
sample_feat = add_features(sample_df.rename(columns={
    "oldbalanceOrg": "oldbalanceOrg", "newbalanceOrig": "newbalanceOrig",
    "oldbalanceDest": "oldbalanceDest", "newbalanceDest": "newbalanceDest"
}))
sample_X = preprocessor.transform(sample_feat[FEATURE_COLS])
sample_proba = model.predict_proba(sample_X)[:, 1]
sample_preds = ["FRAUD" if p >= best_thr else "GENUINE" for p in sample_proba]
print("\n  Sample transaction predictions:")
for row, prob, pred in zip(sample_rows, sample_proba, sample_preds):
    print(f"    {row['transaction_id']} ({row['type']}, ${row['amount']:,.0f}) -> {prob*100:.2f}% -> {pred}")

print("\nDONE. Training complete.")
print(f"\nSaved metrics:")
print(f"  Accuracy:  {acc*100:.3f}%")
print(f"  Precision: {prec*100:.2f}%")
print(f"  Recall:    {rec*100:.2f}%")
print(f"  F1:        {f1*100:.2f}%")
print(f"  ROC-AUC:   {roc*100:.2f}%")
print(f"  PR-AUC:    {prauc*100:.2f}%")
