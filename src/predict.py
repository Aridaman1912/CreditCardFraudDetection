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
            raise KeyError(f"Artifact dictionary is missing key: '{k}'")

    return artifacts


def validate_input_data(
    input_data: Union[Dict[str, Any], pd.DataFrame],
    expected_features: List[str] = REQUIRED_FEATURES
) -> Tuple[bool, str, List[str]]:
    if isinstance(input_data, dict):
        missing = [col for col in expected_features if col not in input_data]
    elif isinstance(input_data, pd.DataFrame):
        missing = [col for col in expected_features if col not in input_data.columns]
    else:
        return False, "Input must be a dictionary or a pandas DataFrame.", []

    if missing:
        return False, f"Missing {len(missing)} required features: {missing[:5]}...", missing

    return True, "All required features are present and valid.", []


def predict_single_transaction(
    features: Union[Dict[str, Any], pd.DataFrame, List[float]],
    artifacts: Dict[str, Any] = None
) -> Dict[str, Any]:
    if artifacts is None:
        artifacts = load_artifacts()

    model = artifacts["model"]
    scaler = artifacts["scaler"]
    feature_names = artifacts["feature_names"]
    threshold = float(artifacts.get("threshold", 0.89))

    if isinstance(features, dict):
        is_valid, msg, missing = validate_input_data(features, feature_names)
        if not is_valid:
            raise ValueError(msg)
        ordered_values = [float(features[col]) for col in feature_names]
        input_df = pd.DataFrame([ordered_values], columns=feature_names)
    elif isinstance(features, pd.DataFrame):
        is_valid, msg, missing = validate_input_data(features, feature_names)
        if not is_valid:
            raise ValueError(msg)
        input_df = features[feature_names].iloc[0:1].astype(float)
    elif isinstance(features, (list, np.ndarray)):
        if len(features) != len(feature_names):
            raise ValueError(f"Expected {len(feature_names)} features, but received {len(features)}.")
        input_df = pd.DataFrame([list(features)], columns=feature_names, dtype=float)
    else:
        raise TypeError("features must be a dict, DataFrame, or list.")

    scaled_array = scaler.transform(input_df)
    probabilities = model.predict_proba(scaled_array)
    fraud_prob = float(probabilities[0, 1])
    fraud_prob_pct = fraud_prob * 100.0

    if fraud_prob >= threshold:
        prediction = "FRAUD"
        is_fraud = True
        status_label = "⚠️ FRAUD DETECTED"
        message = "The transaction has been classified as high risk."
    else:
        prediction = "GENUINE"
        is_fraud = False
        status_label = "✅ GENUINE TRANSACTION"
        message = "The transaction appears to be low risk."

    return {
        "fraud_probability": fraud_prob,
        "fraud_probability_pct": fraud_prob_pct,
        "prediction": prediction,
        "is_fraud": is_fraud,
        "status_label": status_label,
        "message": message,
        "threshold": threshold,
        "scaled_features": scaled_array[0],
        "feature_names": feature_names
    }


def predict_batch(
    df: pd.DataFrame,
    artifacts: Dict[str, Any] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if artifacts is None:
        artifacts = load_artifacts()

    model = artifacts["model"]
    scaler = artifacts["scaler"]
    feature_names = artifacts["feature_names"]
    threshold = float(artifacts.get("threshold", 0.89))

    is_valid, msg, missing = validate_input_data(df, feature_names)
    if not is_valid:
        raise ValueError(f"Batch validation failed: {msg}")

    input_df = df[feature_names].astype(float)
    X_scaled = scaler.transform(input_df)

    probabilities = model.predict_proba(X_scaled)[:, 1]
    predictions = np.where(probabilities >= threshold, "FRAUD", "GENUINE")

    output_df = df.copy()
    output_df["Fraud Probability"] = np.round(probabilities, 4)
    output_df["Fraud Probability (%)"] = np.round(probabilities * 100.0, 2)
    output_df["Prediction"] = predictions

    total_count = len(df)
    fraud_count = int(np.sum(predictions == "FRAUD"))
    genuine_count = total_count - fraud_count
    fraud_rate_pct = (fraud_count / total_count * 100.0) if total_count > 0 else 0.0

    summary = {
        "total_transactions": total_count,
        "flagged_fraud_count": fraud_count,
        "genuine_count": genuine_count,
        "fraud_rate_pct": fraud_rate_pct,
        "threshold_used": threshold
    }

    return output_df, summary
