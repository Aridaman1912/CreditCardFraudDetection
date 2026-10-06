import random
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.predict import (
    DEFAULT_MODEL_PATH,
    REQUIRED_FEATURES,
    TOP_SHAP_FEATURES,
    load_artifacts,
    predict_fraud,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* metric cards */
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

  /* result cards */
  .card-fraud {
    background:#FEF2F2; border:2px solid #DC2626;
    border-radius:14px; padding:28px 32px; margin-top:16px;
    text-align:center;
  }
  .card-genuine {
    background:#F0FDF4; border:2px solid #16A34A;
    border-radius:14px; padding:28px 32px; margin-top:16px;
    text-align:center;
  }
  .verdict-fraud  { font-size:2.4rem; font-weight:900; color:#B91C1C; margin:0; }
  .verdict-gen    { font-size:2.4rem; font-weight:900; color:#15803D; margin:0; }
  .prob-label     { font-size:1rem;   font-weight:500; color:#475569; margin-top:6px; }
  .prob-value     { font-size:1.8rem; font-weight:800; color:#1E293B; margin:0; }
  .threshold-note { font-size:.82rem; color:#94A3B8; margin-top:8px; }
  .truth-badge    { font-size:.85rem; font-weight:600; color:#334155;
                    background:#F1F5F9; border-radius:6px;
                    padding:4px 10px; display:inline-block; margin-top:10px; }

  /* section headings */
  .sh {
    font-size:1.15rem; font-weight:700; color:#1E293B;
    border-bottom:2px solid #E2E8F0;
    padding-bottom:6px; margin:22px 0 12px;
  }
  #MainMenu { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading XGBoost model…")
def get_artifacts():
    return load_artifacts(DEFAULT_MODEL_PATH)

try:
    artifacts = get_artifacts()
    THRESHOLD = float(artifacts.get("threshold", 0.89))
    FEATURE_NAMES = artifacts["feature_names"]
except Exception as exc:
    st.error(f"❌ Failed to load model: {exc}")
    st.stop()


# ── Load sample transactions ──────────────────────────────────────────────────
SAMPLE_FILE = Path("data/sample_transactions.csv")

@st.cache_data(show_spinner=False)
def load_samples():
    if not SAMPLE_FILE.exists():
        return pd.DataFrame(), pd.DataFrame()
    df = pd.read_csv(SAMPLE_FILE)
    genuine = df[df["label"] == "genuine"].reset_index(drop=True)
    fraud   = df[df["label"] == "fraud"].reset_index(drop=True)
    return genuine, fraud

genuine_df, fraud_df = load_samples()


# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💳 Credit Card Fraud Detection")
    st.caption("ML Portfolio Project — XGBoost + SMOTE")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Prediction", "📊 Model Performance", "📈 Explainability", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption(f"Decision threshold: **{THRESHOLD:.2f}**")
    st.caption("Dataset: ULB Credit Card Fraud (2013)")


# ── Helper: run prediction and display result ─────────────────────────────────
def run_and_show(row: dict, ground_truth: str | None = None):
    """Run predict_fraud on a row dict and display a prominent result card."""
    try:
        result = predict_fraud(row, artifacts)
    except Exception as exc:
        st.error(f"Prediction error: {exc}")
        return

    prob_pct = result["fraud_probability_pct"]
    is_fraud = result["is_fraud"]
    verdict  = "🚨 FRAUD" if is_fraud else "✅ GENUINE"
    card_cls = "card-fraud" if is_fraud else "card-genuine"
    v_cls    = "verdict-fraud" if is_fraud else "verdict-gen"

    truth_html = ""
    if ground_truth is not None:
        truth_html = f'<div class="truth-badge">Dataset ground truth: {ground_truth}</div>'

    st.markdown(f"""
    <div class="{card_cls}">
      <div class="prob-label">Fraud Probability</div>
      <div class="prob-value">{prob_pct:.2f}%</div>
      <div class="{v_cls}" style="margin-top:10px;">{verdict}</div>
      <div class="threshold-note">Classification threshold: {THRESHOLD:.2f}</div>
      {truth_html}
    </div>
    """, unsafe_allow_html=True)

    # Gauge chart
    color = "#DC2626" if is_fraud else "#16A34A"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_pct,
        number={"suffix": "%", "font": {"size": 28, "color": color}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "bgcolor": "white",
            "borderwidth": 1,
            "bordercolor": "#CBD5E1",
            "steps": [
                {"range": [0, THRESHOLD * 100], "color": "#F0FDF4"},
                {"range": [THRESHOLD * 100, 100], "color": "#FEF2F2"},
            ],
            "threshold": {
                "line": {"color": "#0F172A", "width": 3},
                "thickness": 0.8,
                "value": THRESHOLD * 100,
            },
        },
    ))
    fig.update_layout(height=190, margin=dict(l=20, r=20, t=5, b=5))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Prediction":
    st.title("💳 Credit Card Fraud Detection")
    st.markdown(
        "Analyze a transaction using our trained **XGBoost** machine learning model "
        "and see the real-time fraud probability."
    )

    st.info(
        "**Note:** This demonstration uses anonymized transaction features from the public "
        "credit-card fraud dataset. The V1–V28 features are hidden from the user because "
        "they are not human-readable transaction fields — they are the result of a PCA "
        "transformation applied to protect cardholder privacy.",
        icon="ℹ️",
    )

    st.markdown("---")

    no_data = genuine_df.empty or fraud_df.empty
    if no_data:
        st.error("Sample transaction file not found. Please ensure `data/sample_transactions.csv` exists.")
        st.stop()

    # ── Three demo buttons ────────────────────────────────────────────────────
    st.markdown('<div class="sh">Choose a demonstration transaction</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    triggered = None     # ("genuine"|"fraud"|"random", row_dict, ground_truth_str)

    with col1:
        if st.button("✅ Analyze Genuine Transaction", use_container_width=True, type="primary"):
            idx = st.session_state.get("gen_idx", 0) % len(genuine_df)
            triggered = ("genuine", genuine_df.iloc[idx].to_dict(), "✅ Genuine")
            st.session_state["gen_idx"] = idx + 1   # cycle through samples

    with col2:
        if st.button("🚨 Analyze Fraudulent Transaction", use_container_width=True, type="primary"):
            idx = st.session_state.get("frd_idx", 0) % len(fraud_df)
            triggered = ("fraud", fraud_df.iloc[idx].to_dict(), "🚨 Fraud")
            st.session_state["frd_idx"] = idx + 1

    with col3:
        if st.button("🎲 Try Random Transaction", use_container_width=True):
            all_df = pd.concat([genuine_df, fraud_df], ignore_index=True)
            row    = all_df.sample(1).iloc[0].to_dict()
            truth  = "✅ Genuine" if row.get("label") == "genuine" else "🚨 Fraud"
            triggered = ("random", row, truth)

    # ── Result ────────────────────────────────────────────────────────────────
    if triggered:
        kind, row, truth = triggered
        st.markdown("---")
        label_map = {"genuine": "Genuine Transaction", "fraud": "Fraudulent Transaction", "random": "Random Transaction"}
        st.markdown(f"**Analyzing:** {label_map.get(kind, 'Transaction')}")
        run_and_show(row, ground_truth=truth)

    elif "last_result" not in st.session_state:
        # Placeholder before first click
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<div style='text-align:center;color:#94A3B8;font-size:1rem;padding:40px 0;'>"
            "Click one of the buttons above to analyze a transaction.</div>",
            unsafe_allow_html=True,
        )

    # ── Pipeline explainer ────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("🔧 How the prediction works (internal pipeline)"):
        st.markdown("""
```
Sample Transaction (real row from dataset)
       ↓
Extract Time + V1–V28 + Amount (hidden from user)
       ↓
Saved StandardScaler (fitted on training data)
       ↓
Saved XGBoost Model (trained with SMOTE)
       ↓
predict_proba() → fraud probability score
       ↓
Apply threshold ≈ 0.89
       ↓
FRAUD if probability ≥ 0.89 else GENUINE
```
The user only sees the **result**. The raw V1–V28 values are used internally by the model
but are never displayed in the user interface.
""")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")
    st.caption(
        "Final results on 56,962 untouched test transactions — "
        f"Decision threshold = {THRESHOLD:.2f}"
    )

    p1, p2, p3 = st.columns(3)
    p1.metric("Accuracy",  "99.956%")
    p2.metric("Precision", "91.95%")
    p3.metric("Recall",    "81.63%")

    p4, p5, p6 = st.columns(3)
    p4.metric("F1 Score", "86.49%")
    p5.metric("ROC-AUC",  "98.11%")
    p6.metric("PR-AUC",   "87.51%")

    st.markdown("---")

    # Confusion matrix heatmap
    st.markdown('<div class="sh">Confusion Matrix</div>', unsafe_allow_html=True)

    cm = [[56857, 7], [18, 80]]
    fig_cm = go.Figure(go.Heatmap(
        z=cm,
        x=["Predicted Genuine", "Predicted Fraud"],
        y=["Actual Genuine", "Actual Fraud"],
        text=[["TN: 56,857<br>(99.99%)", "FP: 7<br>(0.01%)"],
              ["FN: 18<br>(18.37%)", "TP: 80<br>(81.63%)"]],
        texttemplate="%{text}",
        textfont={"size": 14, "color": "white"},
        colorscale="Blues",
        showscale=False,
    ))
    fig_cm.update_layout(
        xaxis=dict(tickfont=dict(size=12)),
        yaxis=dict(tickfont=dict(size=12), autorange="reversed"),
        margin=dict(l=40, r=40, t=20, b=40),
        height=290,
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("True Negatives",  "56,857")
    c2.metric("False Positives", "7")
    c3.metric("False Negatives", "18")
    c4.metric("True Positives",  "80")

    st.markdown("---")

    # Metric explanations
    st.markdown('<div class="sh">Why these metrics matter</div>', unsafe_allow_html=True)
    st.markdown("""
| Metric | What it means |
|--------|--------------|
| **Precision 91.95%** | Of all transactions flagged as fraud, 91.95% were actually fraudulent. |
| **Recall 81.63%** | Of all real fraud transactions, the model caught 81.63%. |
| **F1 Score 86.49%** | Harmonic mean of precision and recall — balances both. |
| **ROC-AUC 98.11%** | Ability to rank fraud above genuine across all thresholds. |
| **PR-AUC 87.51%** | Most meaningful metric for severe class imbalance — prioritizes performance on the minority fraud class. |
""")

    st.markdown("---")

    # Dataset imbalance
    st.markdown('<div class="sh">⚠️ Dataset Imbalance</div>', unsafe_allow_html=True)

    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Total Transactions",  "284,807")
    i2.metric("Genuine",             "284,315")
    i3.metric("Fraudulent",          "492")
    i4.metric("Fraud Rate",          "~0.173%")

    st.markdown("""
A model that always predicts **"genuine"** achieves **99.83% accuracy** while catching **zero frauds**.
This is why **Recall** and **PR-AUC** are the primary metrics, and why the classification threshold
is tuned to **{:.2f}** (not the default 0.50) to maximise F1 on the validation set.
""".format(THRESHOLD))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Explainability":
    st.title("📈 Model Explainability")
    st.markdown(
        "SHAP (SHapley Additive exPlanations) analysis identifies which features "
        "most influence each prediction."
    )
    st.warning(
        "**V1–V28 are anonymized / PCA-transformed features.** "
        "Their individual real-world meanings are not directly interpretable. "
        "We do not claim they represent merchant, location, customer identity, or any specific transaction attribute.",
        icon="⚠️",
    )

    shap_df = pd.DataFrame({
        "Rank":    list(range(1, 11)),
        "Feature": TOP_SHAP_FEATURES,
        "Note":    ["Anonymized PCA feature — high SHAP importance for fraud detection"] * 10,
    })
    st.dataframe(shap_df, use_container_width=True, hide_index=True)

    fig_shap = go.Figure(go.Bar(
        x=list(range(10, 0, -1)),
        y=TOP_SHAP_FEATURES[::-1],
        orientation="h",
        marker=dict(color=list(range(10, 0, -1)), colorscale="Blues", showscale=False),
        text=[f"Rank #{11 - i}" for i in range(1, 11)],
        textposition="outside",
    ))
    fig_shap.update_layout(
        title="Top 10 Features by SHAP Importance",
        xaxis_title="Relative Importance Score",
        yaxis_title="Feature",
        height=380,
        margin=dict(l=40, r=80, t=40, b=40),
    )
    st.plotly_chart(fig_shap, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")

    st.markdown(f"""
This is a **machine learning portfolio project** built on the public
[ULB Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
It is **not** a real banking production system.

---

### 🎯 Objective
Detect fraudulent credit card transactions while handling extreme class imbalance
(only **0.173%** of transactions are fraudulent).

---

### 📦 Dataset
| Property | Value |
|----------|-------|
| Source | ULB Machine Learning Group (Kaggle) |
| Period | September 2013, European cardholders |
| Transactions | 284,807 over 48 hours |
| Fraud cases | 492 (0.173%) |
| Features | `Time`, `V1`–`V28` (PCA-anonymized), `Amount`, `Class` |

---

### 🔧 ML Workflow
1. Stratified train/test split (80/20)
2. `StandardScaler` fitted on training data only
3. **SMOTE** applied to training fold to address class imbalance
4. **XGBoost** trained on the resampled training set
5. Decision threshold tuned to **{THRESHOLD:.2f}** (maximises F1 on validation set)
6. Final evaluation on untouched test set

---

### ⚠️ Important Limitation — Anonymized Features
The V1–V28 features are the result of a **PCA transformation** applied by the original dataset
authors to protect cardholder privacy. Their individual meanings are not disclosed.

Because of this, this application **cannot** accept:
- Merchant name or category
- Card number or customer identity
- Transaction location or bank information

and convert them into V1–V28 to make a prediction.

The deployed application intentionally hides V1–V28 from users and instead provides
**representative demonstration transactions** from the original dataset, run through
the actual trained model.

---

### 🚀 How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

App opens at **http://localhost:8501**.
""")
