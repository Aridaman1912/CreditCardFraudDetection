import io
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from src.predict import (
    load_artifacts,
    predict_fraud,
    predict_dataframe,
    validate_features,
    REQUIRED_FEATURES,
    TOP_SHAP_FEATURES,
    DEFAULT_MODEL_PATH,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* metric cards */
    div[data-testid="stMetric"] {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px 18px;
    }
    div[data-testid="stMetric"] label {
        font-size: 0.78rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: .5px;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.55rem;
        font-weight: 800;
        color: #0F172A;
    }
    /* result cards */
    .res-fraud {
        background: #FEF2F2;
        border: 2px solid #DC2626;
        border-radius: 12px;
        padding: 22px 28px;
        margin-top: 12px;
    }
    .res-genuine {
        background: #F0FDF4;
        border: 2px solid #16A34A;
        border-radius: 12px;
        padding: 22px 28px;
        margin-top: 12px;
    }
    .res-title-fraud  { font-size:1.9rem; font-weight:800; color:#B91C1C; margin:0; }
    .res-title-gen    { font-size:1.9rem; font-weight:800; color:#15803D; margin:0; }
    .res-prob         { font-size:1.1rem; font-weight:600; color:#334155; margin-top:6px; }
    .res-note         { font-size:.88rem; color:#64748B; margin-top:4px; }
    /* section headings */
    .sh { font-size:1.2rem; font-weight:700; color:#1E293B;
          border-bottom:2px solid #E2E8F0; padding-bottom:6px; margin:18px 0 10px; }
    /* number input labels */
    div[data-testid="stNumberInput"] label {
        font-size:.8rem; font-weight:600; color:#334155;
    }
    /* hide streamlit default hamburger */
    #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load model once ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def get_artifacts():
    return load_artifacts(DEFAULT_MODEL_PATH)

try:
    artifacts = get_artifacts()
    THRESHOLD = float(artifacts.get("threshold", 0.89))
    FEATURE_NAMES = artifacts["feature_names"]
except Exception as exc:
    st.error(f"❌ Failed to load model: {exc}")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
SAMPLE_FILE = Path("data/sample_transactions.csv")
BATCH_FILE  = Path("sample_data/batch_test_sample.csv")

with st.sidebar:
    st.markdown("## 💳 Credit Card Fraud Detection")
    st.caption("ML Portfolio Project — XGBoost + SMOTE")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🔍 Prediction", "📁 Batch Prediction", "📊 Model Performance", "🧬 Explainability", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption(f"Model threshold: **{THRESHOLD:.2f}**")
    st.caption("Dataset: ULB Credit Card Fraud (European cardholders, 2013)")


# ── Helper: display prediction result ─────────────────────────────────────────
def show_result(result: dict):
    prob_pct = result["fraud_probability_pct"]
    is_fraud = result["is_fraud"]

    if is_fraud:
        st.markdown(f"""
        <div class="res-fraud">
            <div class="res-title-fraud">🚨 FRAUD</div>
            <div class="res-prob">Fraud Probability: <strong>{prob_pct:.2f}%</strong></div>
            <div class="res-note">
                Probability ≥ threshold ({THRESHOLD:.2f}).
                Prediction is based on the trained XGBoost fraud detection model.
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="res-genuine">
            <div class="res-title-gen">✅ GENUINE</div>
            <div class="res-prob">Fraud Probability: <strong>{prob_pct:.2f}%</strong></div>
            <div class="res-note">
                Probability &lt; threshold ({THRESHOLD:.2f}).
                Prediction is based on the trained XGBoost fraud detection model.
            </div>
        </div>""", unsafe_allow_html=True)

    color = "#DC2626" if is_fraud else "#16A34A"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_pct,
        number={"suffix": "%", "font": {"size": 28, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
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
if page == "🔍 Prediction":
    st.title("💳 Credit Card Fraud Detection")
    st.markdown("**Analyze a transaction and determine whether it is likely to be fraudulent.**")
    st.markdown(
        "> V1–V28 are **anonymized** PCA-transformed features from the original dataset. "
        "Real values are loaded from the sample file or your own CSV upload."
    )
    st.markdown("---")

    # ── Option A: Sample Transactions ─────────────────────────────────────────
    st.markdown('<div class="sh">🔍 Try a Sample Transaction</div>', unsafe_allow_html=True)
    st.caption("These are real rows from the dataset — no values are invented.")

    if not SAMPLE_FILE.exists():
        st.warning("Sample file `data/sample_transactions.csv` not found.")
    else:
        sample_df = pd.read_csv(SAMPLE_FILE)
        genuine_rows = sample_df[sample_df["label"] == "genuine"].reset_index(drop=True)
        fraud_rows   = sample_df[sample_df["label"] == "fraud"].reset_index(drop=True)

        col_a, col_b = st.columns(2)
        run_sample_result = None

        with col_a:
            n_genuine = len(genuine_rows)
            gen_idx = st.selectbox(
                "Genuine transaction sample",
                options=list(range(n_genuine)),
                format_func=lambda i: f"Genuine #{i+1}  (Amount: ${genuine_rows.loc[i,'Amount']:.2f})",
                key="gen_idx",
            )
            if st.button("✅ Try Genuine Transaction", use_container_width=True, type="primary"):
                run_sample_result = ("genuine", gen_idx)

        with col_b:
            n_fraud = len(fraud_rows)
            frd_idx = st.selectbox(
                "Fraudulent transaction sample",
                options=list(range(n_fraud)),
                format_func=lambda i: f"Fraud #{i+1}  (Amount: ${fraud_rows.loc[i,'Amount']:.2f})",
                key="frd_idx",
            )
            if st.button("🚨 Try Fraudulent Transaction", use_container_width=True, type="primary"):
                run_sample_result = ("fraud", frd_idx)

        if run_sample_result:
            kind, idx = run_sample_result
            row_df = genuine_rows if kind == "genuine" else fraud_rows
            transaction_row = row_df.iloc[idx].to_dict()

            st.info(
                f"This is a demonstration transaction from the dataset. "
                f"{'Ground truth: **GENUINE**' if kind == 'genuine' else 'Ground truth: **FRAUD**'}"
            )
            try:
                result = predict_fraud(transaction_row, artifacts)
                show_result(result)
            except Exception as exc:
                st.error(f"Prediction error: {exc}")

    st.markdown("---")

    # ── Option B: CSV Upload ───────────────────────────────────────────────────
    st.markdown('<div class="sh">📁 Upload Transaction CSV</div>', unsafe_allow_html=True)
    st.markdown(
        "Because the dataset uses anonymized V1–V28 features, uploading a CSV is the most practical way "
        "to analyze real transactions. The CSV must contain these columns:"
    )
    st.code("Time, V1, V2, ..., V28, Amount", language="text")

    uploaded = st.file_uploader("Upload a CSV file", type=["csv"], key="single_upload")
    if uploaded:
        try:
            up_df = pd.read_csv(uploaded)
            up_df_clean = up_df.drop(columns=["Class"], errors="ignore")

            is_valid, msg, missing = validate_features(up_df_clean, FEATURE_NAMES)
            if not is_valid:
                st.error(f"❌ {msg}")
            else:
                st.success(f"Loaded {len(up_df_clean):,} row(s). Showing prediction for first row.")
                try:
                    result = predict_fraud(up_df_clean, artifacts)
                    show_result(result)
                except Exception as exc:
                    st.error(f"Prediction error: {exc}")
        except Exception as exc:
            st.error(f"Could not read file: {exc}")

    st.markdown("---")

    # ── Option C: Advanced Manual Input (collapsed) ────────────────────────────
    with st.expander("⚙️ Advanced: Enter Model Features Manually"):
        st.caption(
            "This is an advanced/demo section. V1–V28 are anonymized — "
            "use the sample buttons or CSV upload above for realistic testing."
        )

        if "manual_inputs" not in st.session_state:
            st.session_state["manual_inputs"] = {f: 0.0 for f in FEATURE_NAMES}

        adv_c1, adv_c2, adv_c3 = st.columns(3)
        with adv_c1:
            if st.button("Load Genuine Preset", key="adv_gen"):
                if SAMPLE_FILE.exists():
                    s = pd.read_csv(SAMPLE_FILE)
                    row = s[s["label"] == "genuine"].iloc[0].to_dict()
                    for f in FEATURE_NAMES:
                        st.session_state["manual_inputs"][f] = float(row.get(f, 0.0))
                    st.rerun()
        with adv_c2:
            if st.button("Load Fraud Preset", key="adv_frd"):
                if SAMPLE_FILE.exists():
                    s = pd.read_csv(SAMPLE_FILE)
                    row = s[s["label"] == "fraud"].iloc[0].to_dict()
                    for f in FEATURE_NAMES:
                        st.session_state["manual_inputs"][f] = float(row.get(f, 0.0))
                    st.rerun()
        with adv_c3:
            if st.button("Reset All to 0", key="adv_reset"):
                for f in FEATURE_NAMES:
                    st.session_state["manual_inputs"][f] = 0.0
                st.rerun()

        st.markdown("**Time & Amount**")
        mc1, mc2 = st.columns(2)
        with mc1:
            t_val = st.number_input("Time", value=float(st.session_state["manual_inputs"]["Time"]),
                                    step=100.0, format="%.2f", key="m_Time")
            st.session_state["manual_inputs"]["Time"] = t_val
        with mc2:
            a_val = st.number_input("Amount", value=float(st.session_state["manual_inputs"]["Amount"]),
                                    min_value=0.0, step=10.0, format="%.2f", key="m_Amount")
            st.session_state["manual_inputs"]["Amount"] = a_val

        st.markdown("**V1 – V28**")
        vcols = st.columns(4)
        for i, feat in enumerate([f"V{j}" for j in range(1, 29)]):
            with vcols[i % 4]:
                v = st.number_input(feat, value=float(st.session_state["manual_inputs"][feat]),
                                    step=0.1, format="%.4f", key=f"m_{feat}")
                st.session_state["manual_inputs"][feat] = v

        if st.button("🔍 Predict (Manual Input)", type="primary", use_container_width=True):
            try:
                result = predict_fraud(st.session_state["manual_inputs"], artifacts)
                show_result(result)
            except Exception as exc:
                st.error(f"Prediction error: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: BATCH PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📁 Batch Prediction":
    st.title("📁 Batch Transaction Scoring")
    st.markdown(
        "Upload a CSV containing multiple transactions. All 30 model features are required. "
        "A `Class` column is ignored if present."
    )
    st.code("Time, V1, V2, ..., V28, Amount", language="text")

    col_up, col_demo = st.columns([3, 1])
    with col_up:
        batch_file = st.file_uploader("Upload CSV", type=["csv"], key="batch_upload")
    with col_demo:
        st.markdown("<br>", unsafe_allow_html=True)
        use_demo = st.button("📂 Load Demo (200 rows)", use_container_width=True)

    batch_df = None
    if batch_file:
        try:
            batch_df = pd.read_csv(batch_file)
        except Exception as exc:
            st.error(f"Could not read file: {exc}")
    elif use_demo and BATCH_FILE.exists():
        batch_df = pd.read_csv(BATCH_FILE)
    elif use_demo:
        st.warning("Demo batch file not found.")

    if batch_df is not None:
        st.info(f"Loaded **{len(batch_df):,}** transactions.")
        try:
            out_df, summary = predict_dataframe(batch_df, artifacts)

            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Total",        f"{summary['total']:,}")
            b2.metric("Flagged Fraud", f"{summary['flagged_fraud']:,}")
            b3.metric("Genuine",       f"{summary['genuine']:,}")
            b4.metric("Fraud Rate",    f"{summary['fraud_rate_pct']:.2f}%")

            view_cols = ["Prediction", "Fraud Probability (%)", "Amount", "Time",
                         "V14", "V4", "V12", "V10"]
            view_cols = [c for c in view_cols if c in out_df.columns]

            st.dataframe(
                out_df[view_cols].style.apply(
                    lambda row: ["background-color:#FEF2F2" if row["Prediction"] == "FRAUD" else ""
                                 for _ in row], axis=1
                ).format({"Amount": "${:,.2f}", "Fraud Probability (%)": "{:.2f}%"}),
                use_container_width=True,
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
        except ValueError as ve:
            st.error(f"❌ {ve}")
        except Exception as exc:
            st.error(f"Prediction error: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")
    st.caption("Final untouched test-set results — 56,962 transactions, threshold = 0.89.")

    p1, p2, p3 = st.columns(3)
    p1.metric("Accuracy",  "99.956%")
    p2.metric("Precision", "91.95%")
    p3.metric("Recall",    "81.63%")

    p4, p5, p6 = st.columns(3)
    p4.metric("F1 Score",  "86.49%")
    p5.metric("ROC-AUC",   "98.11%")
    p6.metric("PR-AUC",    "87.51%")

    st.markdown("---")
    st.markdown("**What these metrics mean:**")
    st.markdown("""
| Metric | Meaning |
|--------|---------|
| **Precision** | Of transactions the model flagged as fraud, 91.95% were actually fraud. |
| **Recall** | Of all real fraud transactions, the model detected 81.63%. |
| **F1** | Harmonic mean of precision and recall — balances both. |
| **ROC-AUC** | Overall ability to rank fraud above genuine across all thresholds. |
| **PR-AUC** | Precision-Recall AUC — particularly meaningful for imbalanced datasets like this one. |
""")

    st.markdown("---")
    st.markdown('<div class="sh">Confusion Matrix (Test Set)</div>', unsafe_allow_html=True)

    fig_cm = go.Figure(data=go.Heatmap(
        z=[[56857, 7], [18, 80]],
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
        height=300,
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("True Negatives",  "56,857")
    c2.metric("False Positives", "7")
    c3.metric("False Negatives", "18")
    c4.metric("True Positives",  "80")

    st.markdown("---")
    st.markdown('<div class="sh">⚠️ Why Fraud Detection Is Difficult</div>', unsafe_allow_html=True)
    st.markdown("""
The dataset is **severely imbalanced**:

| | Count |
|---|---|
| Total transactions | 284,807 |
| Fraudulent | 492 |
| Genuine | 284,315 |
| Fraud percentage | **~0.173%** |

A naive model that always predicts "genuine" would achieve **99.83% accuracy** while catching **zero frauds**.
This is why we optimize for **Recall** (catching real fraud) and use **PR-AUC** as the primary evaluation metric,
and why the decision threshold is tuned to **{:.2f}** rather than the default 0.50.
""".format(THRESHOLD))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧬 Explainability":
    st.title("🧬 Model Explainability")
    st.markdown(
        "The following features were identified by **SHAP** (SHapley Additive exPlanations) "
        "analysis as the most impactful for the fraud detection model."
    )
    st.info(
        "**Important:** V1–V28 are **anonymized PCA-transformed features** from the original dataset. "
        "Their individual real-world meanings are not directly interpretable. "
        "We cannot claim they correspond to merchant, location, customer identity, or purchase category."
    )

    shap_df = pd.DataFrame({
        "Rank": range(1, 11),
        "Feature": TOP_SHAP_FEATURES,
        "Note": ["Anonymized PCA feature"] * 10,
    })
    st.dataframe(shap_df, use_container_width=True, hide_index=True)

    fig_shap = go.Figure(go.Bar(
        x=list(range(10, 0, -1)),
        y=TOP_SHAP_FEATURES[::-1],
        orientation="h",
        marker=dict(
            color=list(range(10, 0, -1)),
            colorscale="Blues",
            showscale=False,
        ),
        text=[f"Rank #{11-i}" for i in range(1, 11)],
        textposition="outside",
    ))
    fig_shap.update_layout(
        title="Top 10 Features by SHAP Importance",
        xaxis_title="Relative Importance Score",
        yaxis_title="Feature",
        height=380,
        margin=dict(l=40, r=60, t=40, b=40),
    )
    st.plotly_chart(fig_shap, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("""
This is a **Machine Learning portfolio project** built using the public
[ULB Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

It is **not** a real banking production system. It demonstrates end-to-end ML workflow:
data exploration, handling class imbalance, model training, threshold calibration,
and deployment as an interactive web application.

---

### Dataset
- **Source:** ULB Machine Learning Group
- **Period:** September 2013, European cardholders
- **Size:** 284,807 transactions over 48 hours
- **Features:** `Time`, `V1`–`V28` (PCA-anonymized), `Amount`, `Class`
- **Class imbalance:** only 0.173% fraud

### ML Pipeline
1. Train / validation / test split (stratified)
2. `StandardScaler` fitted on training data only
3. **SMOTE** applied to training fold to address class imbalance
4. **XGBoost** trained on resampled data
5. Decision threshold tuned to **{thr:.2f}** (maximising F1 on validation set)
6. Final evaluation on untouched test set

### V1–V28 Features
These features are the result of a PCA transformation applied by the dataset authors
to protect cardholder privacy. Their original meanings are not disclosed.

### How to run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Uploading your own CSV
The CSV must include: `Time, V1, V2, ..., V28, Amount`
""".format(thr=THRESHOLD))
