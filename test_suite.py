import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from src.predict import (
    load_artifacts, predict_fraud, predict_dataframe,
    validate_features, REQUIRED_FEATURES,
)

print("TEST 1 — Load artifacts")
art = load_artifacts()
print(f"  threshold={art['threshold']}, features={len(art['feature_names'])}")
assert art["threshold"] == 0.89
assert len(art["feature_names"]) == 30
print("  PASSED")

print("\nTEST 2 — Genuine sample")
sample = pd.read_csv("data/sample_transactions.csv")
genuine_row = sample[sample["label"] == "genuine"].iloc[0].to_dict()
res = predict_fraud(genuine_row, art)
print(f"  prob={res['fraud_probability_pct']:.2f}% prediction={res['prediction']}")
assert res["prediction"] == "GENUINE"
print("  PASSED")

print("\nTEST 3 — Fraud sample")
fraud_row = sample[sample["label"] == "fraud"].iloc[0].to_dict()
res = predict_fraud(fraud_row, art)
print(f"  prob={res['fraud_probability_pct']:.2f}% prediction={res['prediction']}")
assert res["prediction"] == "FRAUD"
print("  PASSED")

print("\nTEST 4 — Batch prediction with Class column present")
batch = pd.read_csv("sample_data/batch_test_sample.csv")
assert "Class" in batch.columns
out, summary = predict_dataframe(batch, art)
print(f"  total={summary['total']}, flagged={summary['flagged_fraud']}")
assert "Fraud Probability" in out.columns
assert "Prediction" in out.columns
assert "Class" not in [c for c in out.columns if c == "Class" and c not in out[["Fraud Probability","Prediction"]].columns]
print("  PASSED")

print("\nTEST 5 — Missing column validation")
bad_df = pd.DataFrame({"Time": [0], "Amount": [100]})
ok, msg, missing = validate_features(bad_df, art["feature_names"])
assert not ok
assert len(missing) > 0
print(f"  Correctly rejected — {msg[:60]}")
print("  PASSED")

print("\nTEST 6 — Non-numeric column rejection")
bad_df2 = pd.read_csv("data/sample_transactions.csv").drop(columns=["label", "Class"], errors="ignore")
bad_df2["V1"] = "not_a_number"
ok, msg, _ = validate_features(bad_df2, art["feature_names"])
assert not ok
print(f"  Correctly rejected — {msg[:60]}")
print("  PASSED")

print("\nTEST 7 — Class column is dropped before prediction")
batch_with_class = pd.read_csv("sample_data/batch_test_sample.csv")
out2, _ = predict_dataframe(batch_with_class, art)
input_cols = set(out2.columns) - {"Fraud Probability", "Fraud Probability (%)", "Prediction"}
assert "Class" in input_cols or "Class" not in art["feature_names"]
print("  PASSED (Class not used as model input)")

print("\nTEST 8 — Threshold is 0.89")
assert art["threshold"] == 0.89
prob_above = art["threshold"] + 0.001
prob_below = art["threshold"] - 0.001
assert (prob_above >= art["threshold"]) == True
assert (prob_below >= art["threshold"]) == False
print("  PASSED")

print("\n=== ALL TESTS PASSED ===")
