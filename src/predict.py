from pathlib import Path
from typing import Dict, Any, Tuple, List, Union
import pickle
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PIPELINE_PATH = BASE_DIR / "models" / "paysim_pipeline.pkl"

USER_INPUT_COLS = [
    "type", "amount",
    "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
]

TRANSACTION_TYPES = ["CASH-OUT", "CASH-IN", "PAYMENT", "TRANSFER", "DEBIT"]


def load_pipeline(path: Union[str, Path] = DEFAULT_PIPELINE_PATH) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pipeline artifact not found at {path}")
    with open(path, "rb") as f:
        artifact = pickle.load(f)
    for key in ["preprocessor", "model", "threshold", "feature_cols"]:
        if key not in artifact:
            raise KeyError(f"Artifact missing key: '{key}'")
    return artifact


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["balance_diff_orig"] = df["newbalanceOrig"] - df["oldbalanceOrg"] + df["amount"]
    df["balance_diff_dest"] = df["newbalanceDest"] - df["oldbalanceDest"] - df["amount"]
    df["orig_drained"]      = (df["newbalanceOrig"] == 0).astype(int)
    df["dest_unchanged"]    = (df["newbalanceDest"] == df["oldbalanceDest"]).astype(int)
    df["amount_to_balance"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    return df


def validate_user_csv(df: pd.DataFrame) -> Tuple[bool, str]:
    missing = [c for c in USER_INPUT_COLS if c not in df.columns]
    if missing:
        return False, f"Missing required columns: {missing}"

    invalid_types = df["type"].dropna().unique().tolist()
    bad = [t for t in invalid_types if t not in TRANSACTION_TYPES]
    if bad:
        return False, f"Invalid transaction type(s): {bad}. Allowed: {TRANSACTION_TYPES}"

    num_cols = [c for c in USER_INPUT_COLS if c != "type"]
    for col in num_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                return False, f"Column '{col}' contains non-numeric values."

    if (df["amount"] < 0).any():
        return False, "Column 'amount' must not contain negative values."

    return True, "OK"


def predict_transactions(
    df: pd.DataFrame,
    artifact: Dict[str, Any] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if artifact is None:
        artifact = load_pipeline()

    preprocessor = artifact["preprocessor"]
    model        = artifact["model"]
    threshold    = float(artifact["threshold"])
    feature_cols = artifact["feature_cols"]

    input_df = df[USER_INPUT_COLS].copy()
    if len(input_df) == 0:
        out = df.copy()
        out["Fraud Probability (%)"] = []
        out["Prediction"] = []
        return out, {"total": 0, "flagged_fraud": 0, "genuine": 0,
                     "fraud_rate_pct": 0.0, "threshold": threshold}
    num_cols = [c for c in USER_INPUT_COLS if c != "type"]
    for col in num_cols:
        input_df[col] = pd.to_numeric(input_df[col], errors="coerce").fillna(0)

    enriched = _add_features(input_df)
    X = enriched[feature_cols]
    X_pre = preprocessor.transform(X)

    proba = model.predict_proba(X_pre)[:, 1]
    preds = np.where(proba >= threshold, "FRAUD", "GENUINE")

    out = df.copy()
    out["Fraud Probability (%)"] = np.round(proba * 100, 2)
    out["Prediction"]            = preds

    n_fraud = int((preds == "FRAUD").sum())
    summary = {
        "total":           len(df),
        "flagged_fraud":   n_fraud,
        "genuine":         len(df) - n_fraud,
        "fraud_rate_pct":  round(n_fraud / len(df) * 100, 2) if len(df) > 0 else 0.0,
        "threshold":       threshold,
    }
    return out, summary


def predict_single(row: Dict[str, Any], artifact: Dict[str, Any] = None) -> Dict[str, Any]:
    df = pd.DataFrame([row])
    out, _ = predict_transactions(df, artifact)
    prob     = float(out["Fraud Probability (%)"].iloc[0])
    pred     = out["Prediction"].iloc[0]
    is_fraud = pred == "FRAUD"
    return {
        "fraud_probability_pct": prob,
        "prediction": pred,
        "is_fraud": is_fraud,
        "threshold": artifact["threshold"] if artifact else load_pipeline()["threshold"],
    }
