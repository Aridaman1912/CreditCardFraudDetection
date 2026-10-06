import sys, io
sys.path.insert(0, ".")
import pandas as pd
import numpy as np
from src.predict import (
    load_pipeline, validate_user_csv, predict_transactions,
    predict_single, USER_INPUT_COLS, TRANSACTION_TYPES
)

art = load_pipeline()
print(f"Pipeline loaded. Threshold={art['threshold']}, Features={art['feature_cols']}\n")

# TEST 1 — Single genuine transaction
print("TEST 1 — Single GENUINE transaction")
row = {"type":"PAYMENT","amount":1200.50,"oldbalanceOrg":15000,
       "newbalanceOrig":13799.50,"oldbalanceDest":0,"newbalanceDest":1200.50}
r = predict_single(row, art)
print(f"  Prob={r['fraud_probability_pct']:.2f}% -> {r['prediction']}")
assert r["prediction"] == "GENUINE", f"FAIL: {r}"
print("  PASS\n")

# TEST 2 — Single fraud transaction
print("TEST 2 — Single FRAUD transaction")
row2 = {"type":"TRANSFER","amount":980000,"oldbalanceOrg":980500,
        "newbalanceOrig":0,"oldbalanceDest":1200,"newbalanceDest":1200}
r2 = predict_single(row2, art)
print(f"  Prob={r2['fraud_probability_pct']:.2f}% -> {r2['prediction']}")
assert r2["prediction"] == "FRAUD", f"FAIL: {r2}"
print("  PASS\n")

# TEST 3 — Batch CSV
print("TEST 3 — Batch CSV (sample_user_transactions.csv)")
df = pd.read_csv("data/sample_user_transactions.csv")
ok, msg = validate_user_csv(df)
assert ok, f"Validation failed: {msg}"
out, summary = predict_transactions(df, art)
print(f"  Total={summary['total']}, Fraud={summary['flagged_fraud']}, Genuine={summary['genuine']}")
assert "Prediction" in out.columns
assert "Fraud Probability (%)" in out.columns
for _, row in out.iterrows():
    print(f"    {row.get('transaction_id','?')} ({row['type']}, ${row['amount']:,.0f}) -> {row['Prediction']}")
print("  PASS\n")

# TEST 4 — Missing required column
print("TEST 4 — Missing column validation")
bad_df = pd.DataFrame({"type":["PAYMENT"], "amount":[100]})
ok, msg = validate_user_csv(bad_df)
assert not ok, "Should have failed"
print(f"  Correctly rejected: {msg}")
print("  PASS\n")

# TEST 5 — Invalid transaction type
print("TEST 5 — Invalid transaction type")
bad2 = df.copy()
bad2["type"] = "WIRE-TRANSFER"
ok, msg = validate_user_csv(bad2)
assert not ok
print(f"  Correctly rejected: {msg}")
print("  PASS\n")

# TEST 6 — Negative amount
print("TEST 6 — Negative amount")
bad3 = df.copy()
bad3["amount"] = -500
ok, msg = validate_user_csv(bad3)
assert not ok
print(f"  Correctly rejected: {msg}")
print("  PASS\n")

# TEST 7 — Empty CSV
print("TEST 7 — Empty CSV")
empty = pd.DataFrame(columns=USER_INPUT_COLS)
ok, msg = validate_user_csv(empty)
assert ok, "Empty but valid schema should pass validation"
out_e, summary_e = predict_transactions(empty, art)
assert summary_e["total"] == 0
print("  PASS — empty file handled gracefully\n")

# TEST 8 — No V1-V28 in app.py
print("TEST 8 — No V1-V28 input fields in app.py")
code = open("app.py", encoding="utf-8").read()
# number_input IS present (for amount, balance) but must NOT be labeled V1..V28
import re
# Check no input field is labeled V1, V2...V28
v_inputs = re.findall(r'number_input\(["\']V\d+', code)
assert len(v_inputs) == 0, f"V-feature number_inputs found: {v_inputs}"
# Check the PCA feature names don't appear as UI labels
for v in ["\"V14\"", "\"V28\"", "\"V1,\"", "V1–V28 input"]:
    assert v not in code, f"Found '{v}' in app.py UI"
print("  PASS — No V1-V28 exposed to users\n")

# TEST 9 — Download predictions CSV
print("TEST 9 — Predictions CSV downloadable")
buf = io.StringIO()
out.to_csv(buf, index=False)
csv_str = buf.getvalue()
assert "Prediction" in csv_str
assert "Fraud Probability" in csv_str
print("  PASS — CSV contains Prediction and Fraud Probability columns\n")

# TEST 10 — Pipeline deterministic
print("TEST 10 — Pipeline is deterministic")
r_a = predict_single(row2, art)
r_b = predict_single(row2, art)
assert r_a["fraud_probability_pct"] == r_b["fraud_probability_pct"]
print("  PASS — same input produces same output\n")

# TEST 11 — requirements.txt has needed packages
print("TEST 11 — requirements.txt covers new dependencies")
req = open("requirements.txt", encoding="utf-8").read()
for pkg in ["streamlit", "pandas", "xgboost", "scikit-learn", "plotly", "imbalanced-learn"]:
    assert pkg in req, f"Missing {pkg} in requirements.txt"
print("  PASS\n")

print("=== ALL 11 TESTS PASSED ===")
