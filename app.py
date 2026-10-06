import io
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.predict import (
    DEFAULT_PIPELINE_PATH,
    USER_INPUT_COLS,
    TRANSACTION_TYPES,
    load_pipeline,
    validate_user_csv,
    predict_transactions,
    predict_single,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  div[data-testid="stMetric"] {
    background:#F8FAFC; border:1px solid #E2E8F0;
    border-radius:10px; padding:14px 18px;
  }
  div[data-testid="stMetric"] label {
    font-size:.78rem; font-weight:700; color:#64748B;
    text-transform:uppercase; letter-spacing:.5px;
  }
  div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size:1.55rem; font-weight:800; color:#0F172A;
  }
  .card-fraud {
    background:#FEF2F2; border:2px solid #DC2626;
    border-radius:14px; padding:24px 28px; margin-top:14px; text-align:center;
  }
  .card-genuine {
    background:#F0FDF4; border:2px solid #16A34A;
    border-radius:14px; padding:24px 28px; margin-top:14px; text-align:center;
  }
  .verdict-fraud { font-size:2.2rem; font-weight:900; color:#B91C1C; margin:0; }
  .verdict-gen   { font-size:2.2rem; font-weight:900; color:#15803D; margin:0; }
  .prob-val  { font-size:1.7rem; font-weight:800; color:#1E293B; margin:4px 0; }
  .prob-lbl  { font-size:.9rem;  font-weight:500; color:#475569; }
  .thr-note  { font-size:.8rem;  color:#94A3B8; margin-top:6px; }
  .sh { font-size:1.12rem; font-weight:700; color:#1E293B;
        border-bottom:2px solid #E2E8F0; padding-bottom:5px; margin:20px 0 10px; }
  #MainMenu { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load pipeline ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading fraud detection model…")
def get_artifact():
    return load_pipeline(DEFAULT_PIPELINE_PATH)

try:
    artifact  = get_artifact()
    THRESHOLD = float(artifact["threshold"])
    METRICS   = artifact.get("metrics", {})
except Exception as exc:
    st.error(f"❌ Failed to load model pipeline: {exc}")
    st.stop()

SAMPLE_FILE = Path("data/sample_user_transactions.csv")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💳 Credit Card Fraud Detection")
    st.caption("ML Portfolio · XGBoost + SMOTE · PaySim Schema")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Analyze Transactions", "📊 Model Performance", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption(f"Decision threshold: **{THRESHOLD:.2f}**")
    st.caption("Features: transaction type, amount, account balances")


# ── Shared: result card for single transaction ────────────────────────────────
def show_single_result(result: dict):
    prob     = result["fraud_probability_pct"]
    is_fraud = result["is_fraud"]
    card_cls = "card-fraud" if is_fraud else "card-genuine"
    v_cls    = "verdict-fraud" if is_fraud else "verdict-gen"
    verdict  = "🚨 FRAUD" if is_fraud else "✅ GENUINE"

    st.markdown(f"""
    <div class="{card_cls}">
      <div class="prob-lbl">Fraud Probability</div>
      <div class="prob-val">{prob:.2f}%</div>
      <div class="{v_cls}">{verdict}</div>
      <div class="thr-note">Classification threshold: {THRESHOLD:.2f}</div>
    </div>""", unsafe_allow_html=True)

    color = "#DC2626" if is_fraud else "#16A34A"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        number={"suffix": "%", "font": {"size": 26, "color": color}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "bgcolor": "white",
            "borderwidth": 1, "bordercolor": "#CBD5E1",
            "steps": [
                {"range": [0, THRESHOLD * 100], "color": "#F0FDF4"},
                {"range": [THRESHOLD * 100, 100], "color": "#FEF2F2"},
            ],
            "threshold": {"line": {"color": "#0F172A", "width": 3},
                          "thickness": 0.8, "value": THRESHOLD * 100},
        },
    ))
    fig.update_layout(height=180, margin=dict(l=15, r=15, t=5, b=5))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYZE TRANSACTIONS
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Analyze Transactions":
    st.title("💳 Credit Card Fraud Detection")
    st.markdown(
        "Upload transaction data and let the machine learning model "
        "identify potentially fraudulent transactions."
    )

    tab_batch, tab_single = st.tabs(["📁 Batch Upload (CSV)", "🔍 Single Transaction"])

    # ── TAB 1: BATCH CSV UPLOAD ───────────────────────────────────────────────
    with tab_batch:
        st.markdown('<div class="sh">Upload Transaction Data</div>', unsafe_allow_html=True)
        st.markdown(
            "Upload a CSV file containing transactions. "
            "The file must include these columns:"
        )
        st.code(
            "transaction_id (optional), type, amount, "
            "oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest",
            language="text",
        )
        st.markdown(
            "**Accepted `type` values:** `CASH-OUT` · `CASH-IN` · `PAYMENT` · `TRANSFER` · `DEBIT`"
        )

        col_up, col_dl = st.columns([3, 1])
        with col_up:
            uploaded = st.file_uploader("Browse files", type=["csv"], key="batch_upload")
        with col_dl:
            st.markdown("<br>", unsafe_allow_html=True)
            if SAMPLE_FILE.exists():
                with open(SAMPLE_FILE, "rb") as f:
                    st.download_button(
                        "⬇️ Download Sample CSV",
                        data=f.read(),
                        file_name="sample_user_transactions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

        batch_df = None
        if uploaded:
            try:
                batch_df = pd.read_csv(uploaded)
            except Exception as exc:
                st.error(f"Could not read file: {exc}")

        if batch_df is not None:
            if len(batch_df) == 0:
                st.warning("The uploaded file is empty.")
            else:
                is_valid, msg = validate_user_csv(batch_df)
                if not is_valid:
                    st.error(f"❌ Validation error: {msg}")
                else:
                    st.success(f"✅ Loaded **{len(batch_df):,}** transaction(s) — ready to analyze.")

                    if st.button("🔍 Analyze Transactions", type="primary", use_container_width=True):
                        with st.spinner("Running fraud detection model…"):
                            try:
                                out_df, summary = predict_transactions(batch_df, artifact)

                                s1, s2, s3, s4 = st.columns(4)
                                s1.metric("Total Transactions", f"{summary['total']:,}")
                                s2.metric("Flagged as Fraud",   f"{summary['flagged_fraud']:,}")
                                s3.metric("Genuine",            f"{summary['genuine']:,}")
                                s4.metric("Fraud Rate",         f"{summary['fraud_rate_pct']:.2f}%")

                                st.markdown('<div class="sh">Prediction Results</div>', unsafe_allow_html=True)

                                display_cols = []
                                if "transaction_id" in out_df.columns:
                                    display_cols.append("transaction_id")
                                display_cols += ["type", "amount", "Fraud Probability (%)", "Prediction"]

                                def row_style(row):
                                    color = "#FEF2F2" if row["Prediction"] == "FRAUD" else ""
                                    return [f"background-color:{color}" for _ in row]

                                st.dataframe(
                                    out_df[display_cols].style
                                    .apply(row_style, axis=1)
                                    .format({"amount": "${:,.2f}",
                                             "Fraud Probability (%)": "{:.2f}%"}),
                                    use_container_width=True,
                                )

                                buf = io.StringIO()
                                out_df.to_csv(buf, index=False)
                                st.download_button(
                                    "📥 Download Predictions CSV",
                                    data=buf.getvalue(),
                                    file_name="fraud_predictions.csv",
                                    mime="text/csv",
                                    type="primary",
                                )
                            except Exception as exc:
                                st.error(f"Prediction error: {exc}")

    # ── TAB 2: SINGLE TRANSACTION ─────────────────────────────────────────────
    with tab_single:
        st.markdown('<div class="sh">Enter a Single Transaction</div>', unsafe_allow_html=True)
        st.caption(
            "Fill in the transaction details below. "
            "No anonymized features required — enter the actual transaction information."
        )

        col1, col2 = st.columns(2)
        with col1:
            txn_type  = st.selectbox("Transaction Type", TRANSACTION_TYPES)
            amount    = st.number_input("Amount ($)", min_value=0.0, value=1000.0,
                                        step=100.0, format="%.2f")
            old_orig  = st.number_input("Sender balance BEFORE transaction ($)",
                                        min_value=0.0, value=5000.0, step=100.0, format="%.2f")
            new_orig  = st.number_input("Sender balance AFTER transaction ($)",
                                        min_value=0.0, value=4000.0, step=100.0, format="%.2f")
        with col2:
            old_dest  = st.number_input("Recipient balance BEFORE transaction ($)",
                                        min_value=0.0, value=2000.0, step=100.0, format="%.2f")
            new_dest  = st.number_input("Recipient balance AFTER transaction ($)",
                                        min_value=0.0, value=3000.0, step=100.0, format="%.2f")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔍 Analyze Transaction", type="primary", use_container_width=True):
            row = {
                "type":           txn_type,
                "amount":         amount,
                "oldbalanceOrg":  old_orig,
                "newbalanceOrig": new_orig,
                "oldbalanceDest": old_dest,
                "newbalanceDest": new_dest,
            }
            with st.spinner("Analyzing…"):
                try:
                    result = predict_single(row, artifact)
                    show_single_result(result)
                except Exception as exc:
                    st.error(f"Prediction error: {exc}")

        st.markdown("---")

        with st.expander("💡 How the model works"):
            st.markdown("""
```
User enters: type, amount, account balances
              ↓
Feature engineering:
  · balance_diff_orig  (originator balance change vs amount)
  · balance_diff_dest  (destination balance change vs amount)
  · orig_drained       (was sender's balance drained to 0?)
  · dest_unchanged     (did recipient balance not change?)
  · amount_to_balance  (amount relative to sender's balance)
              ↓
StandardScaler (numeric) + OneHotEncoder (type)
              ↓
Trained XGBoost Classifier
              ↓
predict_proba() → fraud probability score
              ↓
Apply threshold = {thr:.2f}
              ↓
FRAUD if probability ≥ {thr:.2f} else GENUINE
```
**Key fraud signals the model looks for:**
- Sender's balance fully drained after transaction
- Recipient's balance not updated (mule account pattern)
- Large amount relative to sender's available balance
- TRANSFER or CASH-OUT transaction type (fraud rarely occurs in other types)
""".format(thr=THRESHOLD))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")
    st.caption(
        f"Evaluated on untouched test set — {METRICS.get('test_size', ''):,} transactions, "
        f"threshold = {THRESHOLD:.2f}"
    )

    p1, p2, p3 = st.columns(3)
    p1.metric("Accuracy",  f"{METRICS.get('accuracy', 0):.3f}%")
    p2.metric("Precision", f"{METRICS.get('precision', 0):.2f}%")
    p3.metric("Recall",    f"{METRICS.get('recall', 0):.2f}%")

    p4, p5, p6 = st.columns(3)
    p4.metric("F1 Score", f"{METRICS.get('f1', 0):.2f}%")
    p5.metric("ROC-AUC",  f"{METRICS.get('roc_auc', 0):.2f}%")
    p6.metric("PR-AUC",   f"{METRICS.get('pr_auc', 0):.2f}%")

    st.markdown("---")
    st.markdown('<div class="sh">Confusion Matrix</div>', unsafe_allow_html=True)

    cm = METRICS.get("cm", [[0, 0], [0, 0]])
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]

    fig_cm = go.Figure(go.Heatmap(
        z=cm,
        x=["Predicted Genuine", "Predicted Fraud"],
        y=["Actual Genuine", "Actual Fraud"],
        text=[[f"TN: {tn:,}", f"FP: {fp:,}"],
              [f"FN: {fn:,}", f"TP: {tp:,}"]],
        texttemplate="%{text}",
        textfont={"size": 14, "color": "white"},
        colorscale="Blues", showscale=False,
    ))
    fig_cm.update_layout(
        xaxis=dict(tickfont=dict(size=12)),
        yaxis=dict(tickfont=dict(size=12), autorange="reversed"),
        margin=dict(l=40, r=40, t=20, b=40), height=280,
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("True Negatives",  f"{tn:,}")
    c2.metric("False Positives", f"{fp:,}")
    c3.metric("False Negatives", f"{fn:,}")
    c4.metric("True Positives",  f"{tp:,}")

    st.markdown("---")
    st.markdown('<div class="sh">Why these metrics matter</div>', unsafe_allow_html=True)
    st.markdown("""
| Metric | Why it matters |
|--------|----------------|
| **Precision** | Of transactions flagged as fraud, what fraction were actually fraudulent? |
| **Recall** | Of all real fraud, what fraction did the model catch? |
| **F1** | Harmonic mean — balanced measure when both precision and recall matter. |
| **PR-AUC** | Most important for imbalanced datasets: area under precision-recall curve. |
| **ROC-AUC** | General discrimination ability across all thresholds. |

> A naive model that always predicts "genuine" achieves high accuracy (~99.85%) while catching
> **zero fraud transactions**. This is why Recall and PR-AUC are the primary evaluation metrics.
""")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown(f"""
This is a **machine learning portfolio project** demonstrating end-to-end fraud detection
with user-friendly, interpretable transaction features.

---

### 🎯 Problem Statement
Detect fraudulent financial transactions in a severely imbalanced dataset
where fraud represents only ~0.15% of all transactions.

---

### 📦 Dataset Schema (PaySim-compatible)
The model is trained on a synthetic dataset following the PaySim mobile money transaction schema:

| Feature | Description |
|---------|-------------|
| `type` | Transaction type: CASH-OUT, CASH-IN, PAYMENT, TRANSFER, DEBIT |
| `amount` | Transaction amount ($) |
| `oldbalanceOrg` | Sender's account balance before the transaction |
| `newbalanceOrig` | Sender's account balance after the transaction |
| `oldbalanceDest` | Recipient's account balance before the transaction |
| `newbalanceDest` | Recipient's account balance after the transaction |

---

### 🔧 ML Pipeline
1. Stratified train/validation/test split (60/20/20)
2. Feature engineering (balance differences, drain detection, amount-to-balance ratio)
3. `StandardScaler` + `OneHotEncoder` via `ColumnTransformer` — fitted on training data only
4. **SMOTE** applied to training data only to address class imbalance
5. **XGBoost** trained on resampled data
6. Threshold tuned on validation set to maximise F1 (threshold = **{THRESHOLD:.2f}**)
7. Final evaluation on untouched test set

---

### ⚠️ Why not the original ULB Credit Card dataset?
The widely-known ULB dataset uses **V1–V28** features, which are the result of a PCA
transformation applied by the dataset authors to protect cardholder privacy.
These features are not interpretable or reproducible from normal transaction data.

This project deliberately uses a dataset with **human-readable features** so that
the deployed application can accept real transaction information from users —
rather than requiring them to enter mathematical principal components.

---

### 🚀 How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

### 📁 Input CSV Format
```csv
transaction_id,type,amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest
TX001,PAYMENT,1200.50,15000,13799.50,0,1200.50
TX002,TRANSFER,980000,980500,0,1200,1200
```
`transaction_id` is optional. Download the sample CSV from the app for a template.
""")
