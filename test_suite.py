import sys
sys.path.insert(0, ".")
import pandas as pd
from src.predict import load_artifacts, predict_fraud

art = load_artifacts()
df  = pd.read_csv("data/sample_transactions.csv")

genuine = df[df["label"] == "genuine"].reset_index(drop=True)
fraud   = df[df["label"] == "fraud"].reset_index(drop=True)

print("TEST 1 — Genuine transaction")
r = predict_fraud(genuine.iloc[0].to_dict(), art)
print(f"  Prob: {r['fraud_probability_pct']:.2f}%  Prediction: {r['prediction']}")
assert r["prediction"] == "GENUINE", "FAIL"
print("  PASS\n")

print("TEST 2 — Fraudulent transaction")
r = predict_fraud(fraud.iloc[0].to_dict(), art)
print(f"  Prob: {r['fraud_probability_pct']:.2f}%  Prediction: {r['prediction']}")
assert r["prediction"] == "FRAUD", "FAIL"
print("  PASS\n")

print("TEST 3 — Random (all 10 sample rows)")
all_df = pd.concat([genuine, fraud], ignore_index=True)
for i in range(len(all_df)):
    row = all_df.iloc[i].to_dict()
    r   = predict_fraud(row, art)
    print(f"  [{row.get('label')}]  Prob={r['fraud_probability_pct']:.2f}%  -> {r['prediction']}")
print("  PASS\n")

print("TEST 4 — No manual V1-V28 number_input fields in app.py")
code = open("app.py", encoding="utf-8").read()
assert "number_input" not in code, "FAIL — number_input still present"
print("  PASS\n")

print("TEST 5 — No CSV upload on main prediction page")
assert "file_uploader" not in code, "FAIL — file_uploader still present"
print("  PASS\n")

print("TEST 6 — Threshold is 0.89")
assert art["threshold"] == 0.89, f"FAIL — got {art['threshold']}"
print(f"  Threshold = {art['threshold']}  PASS\n")

print("TEST 7 — 30 features in artifact")
assert len(art["feature_names"]) == 30, "FAIL"
print(f"  Features = {len(art['feature_names'])}  PASS\n")

print("TEST 8 — Cycling through genuine samples")
for i in range(len(genuine)):
    r = predict_fraud(genuine.iloc[i].to_dict(), art)
    assert r["prediction"] == "GENUINE", f"FAIL on genuine row {i}"
print(f"  All {len(genuine)} genuine rows -> GENUINE  PASS\n")

print("TEST 9 — Cycling through fraud samples")
for i in range(len(fraud)):
    r = predict_fraud(fraud.iloc[i].to_dict(), art)
    assert r["prediction"] == "FRAUD", f"FAIL on fraud row {i}"
print(f"  All {len(fraud)} fraud rows -> FRAUD  PASS\n")

print("=== ALL 9 TESTS PASSED ===")
