from pathlib import Path
from typing import Dict, Any, Tuple, List, Union
import pickle
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "fraud_detection_model.pkl"

REQUIRED_FEATURES = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
]

TOP_SHAP_FEATURES = [
    "V14", "V4", "V8", "V12", "V10", "V1", "V18", "V3", "V11", "V22"
]


def load_artifacts(model_path: Union[str, Path] = DEFAULT_MODEL_PATH) -> Dict[str, Any]:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}.")

    with open(model_path, "rb") as f:
        artifacts = pickle.load(f)

    for k in ["model", "scaler", "feature_names", "threshold"]:
        if k not in artifacts:
            raise KeyError(f"Artifact is missing key: '{k}'")

    return artifacts


def validate_features(
    input_data: Union[Dict[str, Any], pd.DataFrame],
    feature_names: List[str]
) -> Tuple[bool, str, List[str]]:
    if isinstance(input_data, dict):
        cols = list(input_data.keys())
    elif isinstance(input_data, pd.DataFrame):
        cols = list(input_data.columns)
    else:
        return False, "Input must be a dict or DataFrame.", []

    missing = [f for f in feature_names if f not in cols]
    if missing:
        return False, f"Missing required columns: {missing}", missing

    if isinstance(input_data, pd.DataFrame):
        for f in feature_names:
            if not pd.api.types.is_numeric_dtype(input_data[f]):
                return False, f"Column '{f}' contains non-numeric values.", [f]

    return True, "OK", []


def predict_fraud(
    transaction: Union[Dict[str, Any], pd.DataFrame],
    artifacts: Dict[str, Any] = None
) -> Dict[str, Any]:
    if artifacts is None:
        artifacts = load_artifacts()

    model = artifacts["model"]
    scaler = artifacts["scaler"]
    feature_names = artifacts["feature_names"]
    threshold = float(artifacts.get("threshold", 0.89))

    if isinstance(transaction, dict):
        is_valid, msg, _ = validate_features(transaction, feature_names)
        if not is_valid:
            raise ValueError(msg)
        ordered = [float(transaction[col]) for col in feature_names]
        input_df = pd.DataFrame([ordered], columns=feature_names)
    elif isinstance(transaction, pd.DataFrame):
        is_valid, msg, _ = validate_features(transaction, feature_names)
        if not is_valid:
            raise ValueError(msg)
        input_df = transaction[feature_names].iloc[0:1].astype(float)
    elif isinstance(transaction, (list, np.ndarray)):
        if len(transaction) != len(feature_names):
            raise ValueError(f"Expected {len(feature_names)} features, got {len(transaction)}.")
        input_df = pd.DataFrame([list(transaction)], columns=feature_names, dtype=float)
    else:
        raise TypeError("transaction must be a dict, DataFrame, or list.")

    scaled = scaler.transform(input_df)
    proba = model.predict_proba(scaled)
    fraud_prob = float(proba[0, 1])

    is_fraud = fraud_prob >= threshold
    return {
        "fraud_probability": fraud_prob,
        "fraud_probability_pct": fraud_prob * 100.0,
        "prediction": "FRAUD" if is_fraud else "GENUINE",
        "is_fraud": is_fraud,
        "threshold": threshold,
    }


def predict_dataframe(
    df: pd.DataFrame,
    artifacts: Dict[str, Any] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if artifacts is None:
        artifacts = load_artifacts()

    model = artifacts["model"]
    scaler = artifacts["scaler"]
    feature_names = artifacts["feature_names"]
    threshold = float(artifacts.get("threshold", 0.89))

    df = df.drop(columns=["Class"], errors="ignore")

    is_valid, msg, missing = validate_features(df, feature_names)
    if not is_valid:
        raise ValueError(msg)

    input_df = df[feature_names].astype(float)
    X_scaled = scaler.transform(input_df)

    proba = model.predict_proba(X_scaled)[:, 1]
    preds = np.where(proba >= threshold, "FRAUD", "GENUINE")

    out = df.copy()
    out["Fraud Probability"] = np.round(proba, 4)
    out["Fraud Probability (%)"] = np.round(proba * 100.0, 2)
    out["Prediction"] = preds

    n_fraud = int(np.sum(preds == "FRAUD"))
    summary = {
        "total": len(df),
        "flagged_fraud": n_fraud,
        "genuine": len(df) - n_fraud,
        "fraud_rate_pct": (n_fraud / len(df) * 100.0) if len(df) > 0 else 0.0,
        "threshold": threshold,
    }

    return out, summary


predict_single_transaction = predict_fraud
predict_batch = predict_dataframe
