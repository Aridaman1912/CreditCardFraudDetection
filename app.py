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

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme-Adaptive Minimal Styling ───────────────────────────────────────────
st.markdown("""
<style>
  /* Theme-adaptive metrics cards */
  div[data-testid="stMetric"] {
    background: rgba(128, 128, 128, 0.08);
    border: 1px solid rgba(128, 128, 128, 0.18);
    border-radius: 10px;
    padding: 10px 14px;
  }
  div[data-testid="stMetric"] label {
    font-size: 0.75rem;
    font-weight: 600;
    opacity: 0.75;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 1.45rem;
    font-weight: 700;
  }

  /* Result banners */
  .res-banner-fraud {
    background: rgba(220, 38, 38, 0.12);
    border: 1.5px solid #DC2626;
    border-radius: 12px;
    padding: 18px 24px;
    margin: 12px 0;
    text-align: center;
  }
  .res-banner-gen {
    background: rgba(22, 163, 74, 0.12);
    border: 1.5px solid #16A34A;
    border-radius: 12px;
    padding: 18px 24px;
    margin: 12px 0;
    text-align: center;
  }
  .res-title-fraud { font-size: 1.8rem; font-weight: 800; color: #EF4444; margin: 0; }
  .res-title-gen   { font-size: 1.8rem; font-weight: 800; color: #22C55E; margin: 0; }
  .res-prob        { font-size: 1.15rem; font-weight: 600; margin: 4px 0 0 0; }
  .res-sub         { font-size: 0.8rem; opacity: 0.7; margin-top: 4px; }

  /* Compact section headers */
  .sub-head {
    font-size: 1.05rem;
    font-weight: 600;
    margin: 16px 0 8px;
    opacity: 0.9;
  }
  #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load Model Pipeline ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def get_artifact():
    return load_pipeline(DEFAULT_PIPELINE_PATH)

try:
    artifact  = get_artifact()
    THRESHOLD = float(artifact["threshold"])
    METRICS   = artifact.get("metrics", {})
except Exception as exc:
    st.error(f"Failed to load model pipeline: {exc}")
    st.stop()

SAMPLE_FILE = Path("data/sample_user_transactions.csv")


# ── Clean Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💳 Fraud Guard")
    page = st.radio(
        "Navigation",
        ["🔍 Analyze", "📊 Performance", "ℹ️ About"],
        label_visibility="collapsed",
    )


# ── Shared: Result Card ───────────────────────────────────────────────────────
def show_single_result(result: dict):
    prob     = result["fraud_probability_pct"]
    is_fraud = result["is_fraud"]
    cls_name = "res-banner-fraud" if is_fraud else "res-banner-gen"
    title_cls = "res-title-fraud" if is_fraud else "res-title-gen"
    verdict   = "🚨 FRAUD DETECTED" if is_fraud else "✅ GENUINE TRANSACTION"

    st.markdown(f"""
    <div class="{cls_name}">
      <div class="{title_cls}">{verdict}</div>
      <div class="res-prob">Risk Probability: <strong>{prob:.2f}%</strong></div>
      <div class="res-sub">Decision threshold: {THRESHOLD:.2f}</div>
    </div>""", unsafe_allow_html=True)

    color = "#DC2626" if is_fraud else "#16A34A"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        number={"suffix": "%", "font": {"size": 26, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "bgcolor": "rgba(128, 128, 128, 0.1)",
            "borderwidth": 1,
            "bordercolor": "rgba(128, 128, 128, 0.2)",
            "steps": [
                {"range": [0, THRESHOLD * 100], "color": "rgba(22, 163, 74, 0.12)"},
                {"range": [THRESHOLD * 100, 100], "color": "rgba(220, 38, 38, 0.12)"},
            ],
            "threshold": {
                "line": {"color": "#38BDF8", "width": 2},
                "thickness": 0.8,
                "value": THRESHOLD * 100
            },
        },
    ))
    fig.update_layout(
        height=170,
        margin=dict(l=20, r=20, t=5, b=5),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "gray"},
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. ANALYZE PAGE
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Analyze":
    st.title("Credit Card Fraud Detection")
    st.caption("AI-powered transaction risk scoring")

    tab_batch, tab_single = st.tabs(["📁 Batch Upload", "⚡ Single Transaction"])

    # ── TAB 1: BATCH CSV ──────────────────────────────────────────────────────
    with tab_batch:
        col_up, col_btn = st.columns([3, 1])
        with col_up:
            uploaded = st.file_uploader(
                "Upload Transaction CSV",
                type=["csv"],
                label_visibility="collapsed",
                key="batch_upload"
            )
        with col_btn:
            if SAMPLE_FILE.exists():
                with open(SAMPLE_FILE, "rb") as f:
                    st.download_button(
                        "⬇️ Download Template",
                        data=f.read(),
                        file_name="sample_user_transactions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

        with st.expander("ℹ️ Supported CSV Format", expanded=False):
            st.markdown("""
            **Required columns:** `type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`  
            *(Optional: `transaction_id`)*  
            **Types:** `CASH-OUT`, `CASH-IN`, `PAYMENT`, `TRANSFER`, `DEBIT`
            """)

        batch_df = None
        if uploaded:
            try:
                batch_df = pd.read_csv(uploaded)
            except Exception as exc:
                st.error(f"Error reading CSV: {exc}")

        if batch_df is not None:
            if len(batch_df) == 0:
                st.warning("Uploaded file is empty.")
            else:
                is_valid, msg = validate_user_csv(batch_df)
                if not is_valid:
                    st.error(f"❌ {msg}")
                else:
                    if st.button("🔍 Analyze Transactions", type="primary", use_container_width=True):
                        with st.spinner("Analyzing transactions…"):
                            try:
                                out_df, summary = predict_transactions(batch_df, artifact)

                                s1, s2, s3, s4 = st.columns(4)
                                s1.metric("Total", f"{summary['total']:,}")
                                s2.metric("Flagged Fraud", f"{summary['flagged_fraud']:,}")
                                s3.metric("Genuine", f"{summary['genuine']:,}")
                                s4.metric("Fraud Rate", f"{summary['fraud_rate_pct']:.1f}%")

                                st.markdown('<div class="sub-head">Results</div>', unsafe_allow_html=True)

                                display_cols = []
                                if "transaction_id" in out_df.columns:
                                    display_cols.append("transaction_id")
                                display_cols += ["type", "amount", "Fraud Probability (%)", "Prediction"]

                                def row_style(row):
                                    color = "rgba(220, 38, 38, 0.15)" if row["Prediction"] == "FRAUD" else ""
                                    return [f"background-color: {color}" for _ in row]

                                st.dataframe(
                                    out_df[display_cols].style
                                    .apply(row_style, axis=1)
                                    .format({"amount": "${:,.2f}",
                                             "Fraud Probability (%)": "{:.2f}%"}),
                                    use_container_width=True,
                                    height=300,
                                )

                                buf = io.StringIO()
                                out_df.to_csv(buf, index=False)
                                st.download_button(
                                    "📥 Download Results CSV",
                                    data=buf.getvalue(),
                                    file_name="fraud_predictions.csv",
                                    mime="text/csv",
                                    type="primary",
                                )
                            except Exception as exc:
                                st.error(f"Error during prediction: {exc}")

    # ── TAB 2: SINGLE TRANSACTION ─────────────────────────────────────────────
    with tab_single:
        col1, col2 = st.columns(2)
        with col1:
            txn_type = st.selectbox("Transaction Type", TRANSACTION_TYPES)
            amount   = st.number_input("Amount ($)", min_value=0.0, value=1200.0, step=50.0, format="%.2f")
            old_orig = st.number_input("Sender Balance Before ($)", min_value=0.0, value=5000.0, step=100.0, format="%.2f")
            new_orig = st.number_input("Sender Balance After ($)", min_value=0.0, value=3800.0, step=100.0, format="%.2f")
        with col2:
            st.markdown("<div style='height: 72px'></div>", unsafe_allow_html=True)
            old_dest = st.number_input("Recipient Balance Before ($)", min_value=0.0, value=1000.0, step=100.0, format="%.2f")
            new_dest = st.number_input("Recipient Balance After ($)", min_value=0.0, value=2200.0, step=100.0, format="%.2f")

        if st.button("Check Transaction Risk", type="primary", use_container_width=True):
            row = {
                "type":           txn_type,
                "amount":         amount,
                "oldbalanceOrg":  old_orig,
                "newbalanceOrig": new_orig,
                "oldbalanceDest": old_dest,
                "newbalanceDest": new_dest,
            }
            try:
                result = predict_single(row, artifact)
                show_single_result(result)
            except Exception as exc:
                st.error(f"Error: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. PERFORMANCE PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Performance":
    st.title("Model Performance")
    st.caption(f"Evaluated on {METRICS.get('test_size', 40000):,} test transactions · Decision Threshold = {THRESHOLD:.2f}")

    p1, p2, p3 = st.columns(3)
    p1.metric("Accuracy",  f"{METRICS.get('accuracy', 99.99):.3f}%")
    p2.metric("Precision", f"{METRICS.get('precision', 100.0):.2f}%")
    p3.metric("Recall",    f"{METRICS.get('recall', 93.33):.2f}%")

    p4, p5, p6 = st.columns(3)
    p4.metric("F1 Score", f"{METRICS.get('f1', 96.55):.2f}%")
    p5.metric("ROC-AUC",  f"{METRICS.get('roc_auc', 99.99):.2f}%")
    p6.metric("PR-AUC",   f"{METRICS.get('pr_auc', 96.63):.2f}%")

    st.markdown('<div class="sub-head">Confusion Matrix</div>', unsafe_allow_html=True)

    cm = METRICS.get("cm", [[39940, 0], [4, 56]])
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]

    fig_cm = go.Figure(go.Heatmap(
        z=cm,
        x=["Predicted Genuine", "Predicted Fraud"],
        y=["Actual Genuine", "Actual Fraud"],
        text=[[f"TN: {tn:,}", f"FP: {fp:,}"],
              [f"FN: {fn:,}", f"TP: {tp:,}"]],
        texttemplate="%{text}",
        textfont={"size": 14},
        colorscale="Blues",
        showscale=False,
    ))
    fig_cm.update_layout(
        xaxis=dict(tickfont=dict(size=12)),
        yaxis=dict(tickfont=dict(size=12), autorange="reversed"),
        margin=dict(l=30, r=30, t=15, b=30),
        height=260,
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "gray"},
    )
    st.plotly_chart(fig_cm, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 3. ABOUT PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("About This Project")
    st.markdown("""
This application detects fraudulent financial transactions using an end-to-end Machine Learning pipeline.

### Highlights
- **Model:** XGBoost Classifier trained with SMOTE class imbalance compensation.
- **Interpretable Features:** Operates on standard transaction attributes (type, amount, balance changes before and after).
- **Threshold Calibration:** Optimal F1-score threshold tuned on validation data ($\approx 0.25$).
- **Deployment:** Powered by native XGBoost JSON serialization for high reliability.

*Note: This is a portfolio demonstration project.*
""")
