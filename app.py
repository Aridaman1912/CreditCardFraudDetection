from pathlib import Path
import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from src.predict import (
    load_artifacts,
    predict_single_transaction,
    predict_batch,
    REQUIRED_FEATURES,
    TOP_SHAP_FEATURES,
    DEFAULT_MODEL_PATH
)

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 12px 16px;
        border-radius: 8px;
    }
    div[data-testid="stMetric"] label {
        font-size: 0.82rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0F172A;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #E2E8F0;
    }
    .result-card-fraud {
        background-color: #FEF2F2;
        border: 2px solid #DC2626;
        border-radius: 10px;
        padding: 20px 24px;
        margin: 15px 0;
    }
    .result-card-genuine {
        background-color: #F0FDF4;
        border: 2px solid #16A34A;
        border-radius: 10px;
        padding: 20px 24px;
        margin: 15px 0;
    }
    .result-title-fraud {
        font-size: 1.8rem;
        font-weight: 800;
        color: #B91C1C;
        margin: 0;
    }
    .result-title-genuine {
        font-size: 1.8rem;
        font-weight: 800;
        color: #15803D;
        margin: 0;
    }
    .result-sub {
        font-size: 1.15rem;
        font-weight: 600;
        color: #334155;
        margin-top: 6px;
    }
    .result-msg {
        font-size: 0.95rem;
        color: #64748B;
        margin-top: 4px;
    }
    div[data-testid="stNumberInput"] label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #334155;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading trained XGBoost model and scaler...")
def get_artifacts():
    return load_artifacts(DEFAULT_MODEL_PATH)


try:
    artifacts = get_artifacts()
    threshold = float(artifacts.get("threshold", 0.89))
except Exception as e:
    st.error(f"Failed to load model artifact: {e}")
    st.stop()

st.title("Credit Card Fraud Detection")
st.markdown("##### AI-powered transaction risk prediction using XGBoost")
st.markdown("Enter the transaction parameters below to compute real-time fraud probability and risk classification.")

st.markdown("---")

fraud_samples_file = Path("sample_data/fraud_samples.csv")
legit_samples_file = Path("sample_data/legitimate_samples.csv")

if "feature_inputs" not in st.session_state:
    st.session_state["feature_inputs"] = {col: 0.0 for col in REQUIRED_FEATURES}
    st.session_state["feature_inputs"]["Time"] = 1000.0
    st.session_state["feature_inputs"]["Amount"] = 45.00

col_preset_label, col_btn_gen, col_btn_frd, col_btn_reset = st.columns([1.5, 1.2, 1.2, 1])

with col_preset_label:
    st.markdown("**Quick Test Presets:**")

with col_btn_gen:
    if st.button("✅ Load Genuine Sample", use_container_width=True):
        if legit_samples_file.exists():
            df_legit = pd.read_csv(legit_samples_file)
            row = df_legit.iloc[0].to_dict()
            for col in REQUIRED_FEATURES:
                st.session_state["feature_inputs"][col] = float(row.get(col, 0.0))
            st.rerun()
        else:
            for col in REQUIRED_FEATURES:
                st.session_state["feature_inputs"][col] = 0.0
            st.session_state["feature_inputs"]["Time"] = 1200.0
            st.session_state["feature_inputs"]["Amount"] = 35.50
            st.rerun()

with col_btn_frd:
    if st.button("⚠️ Load Fraud Sample", use_container_width=True):
        if fraud_samples_file.exists():
            df_fraud = pd.read_csv(fraud_samples_file)
            row = df_fraud.iloc[0].to_dict()
            for col in REQUIRED_FEATURES:
                st.session_state["feature_inputs"][col] = float(row.get(col, 0.0))
            st.rerun()
        else:
            st.session_state["feature_inputs"]["V14"] = -5.8
            st.session_state["feature_inputs"]["V12"] = -4.2
            st.session_state["feature_inputs"]["V10"] = -3.9
            st.session_state["feature_inputs"]["V4"] = 3.5
            st.session_state["feature_inputs"]["Amount"] = 280.0
            st.rerun()

with col_btn_reset:
    if st.button("🔄 Reset to 0", use_container_width=True):
        for col in REQUIRED_FEATURES:
            st.session_state["feature_inputs"][col] = 0.0
        st.session_state["feature_inputs"]["Time"] = 0.0
        st.session_state["feature_inputs"]["Amount"] = 0.0
        st.rerun()

st.markdown('<div class="section-header">💳 Transaction Details</div>', unsafe_allow_html=True)

col_time, col_amount = st.columns(2)
with col_time:
    time_val = st.number_input(
        "Time (Seconds elapsed from first transaction)",
        value=float(st.session_state["feature_inputs"].get("Time", 0.0)),
        step=100.0,
        format="%.2f",
        key="input_Time"
    )
    st.session_state["feature_inputs"]["Time"] = time_val

with col_amount:
    amount_val = st.number_input(
        "Amount ($ USD)",
        value=float(st.session_state["feature_inputs"].get("Amount", 0.0)),
        min_value=0.0,
        step=10.0,
        format="%.2f",
        key="input_Amount"
    )
    st.session_state["feature_inputs"]["Amount"] = amount_val

st.markdown("##### PCA Engineered Features (V1 – V28)")

pca_cols = [f"V{i}" for i in range(1, 29)]
cols_ui = st.columns(4)

for idx, feature_name in enumerate(pca_cols):
    with cols_ui[idx % 4]:
        val = st.number_input(
            feature_name,
            value=float(st.session_state["feature_inputs"].get(feature_name, 0.0)),
            step=0.1,
            format="%.4f",
            key=f"input_{feature_name}"
        )
        st.session_state["feature_inputs"][feature_name] = val

st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("🔍 Predict Transaction", type="primary", use_container_width=True)

if predict_clicked:
    try:
        result = predict_single_transaction(st.session_state["feature_inputs"], artifacts)
        prob = result["fraud_probability"]
        prob_pct = result["fraud_probability_pct"]
        is_fraud = result["is_fraud"]

        st.markdown('<div class="section-header">📊 Prediction Result</div>', unsafe_allow_html=True)

        if is_fraud:
            st.markdown(f"""
            <div class="result-card-fraud">
                <div class="result-title-fraud">⚠️ FRAUD DETECTED</div>
                <div class="result-sub">Fraud Probability: {prob_pct:.2f}%</div>
                <div class="result-msg">{result['message']} (Decision threshold: {threshold:.2f})</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card-genuine">
                <div class="result-title-genuine">✅ GENUINE TRANSACTION</div>
                <div class="result-sub">Fraud Probability: {prob_pct:.2f}%</div>
                <div class="result-msg">{result['message']} (Decision threshold: {threshold:.2f})</div>
            </div>
            """, unsafe_allow_html=True)

        bar_color = "#DC2626" if is_fraud else "#16A34A"
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_pct,
            number={'suffix': "%", 'font': {'size': 30, 'color': bar_color}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
                'bar': {'color': bar_color},
                'bgcolor': "white",
                'borderwidth': 1,
                'bordercolor': "#CBD5E1",
                'steps': [
                    {'range': [0, threshold * 100], 'color': "#F0FDF4"},
                    {'range': [threshold * 100, 100], 'color': "#FEF2F2"}
                ],
                'threshold': {
                    'line': {'color': "#0F172A", 'width': 3},
                    'thickness': 0.8,
                    'value': threshold * 100
                }
            }
        ))
        fig_gauge.update_layout(
            height=200,
            margin=dict(l=30, r=30, t=10, b=10)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    except Exception as err:
        st.error(f"Error during prediction: {err}")

st.markdown("---")

tab_perf, tab_cm, tab_batch, tab_info, tab_explain = st.tabs([
    "📈 Model Performance",
    "🎯 Confusion Matrix",
    "📁 Batch Prediction",
    "ℹ️ Model Information",
    "🔍 Feature Explainability"
])

with tab_perf:
    st.markdown('<div class="section-header">Final Test-Set Performance</div>', unsafe_allow_html=True)
    st.caption("Performance evaluated on 56,962 untouched test transactions using decision threshold = 0.89.")

    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", "99.956%")
    m2.metric("Precision", "91.95%")
    m3.metric("Recall", "81.63%")

    m4, m5, m6 = st.columns(3)
    m4.metric("F1 Score", "86.49%")
    m5.metric("ROC-AUC", "98.11%")
    m6.metric("PR-AUC", "87.51%")

with tab_cm:
    st.markdown('<div class="section-header">Final Test Confusion Matrix</div>', unsafe_allow_html=True)
    st.caption("Decision Threshold = 0.89 | Total Test Transactions = 56,962")

    cm_data = [[56857, 7], [18, 80]]

    fig_cm = go.Figure(data=go.Heatmap(
        z=cm_data,
        x=["Predicted Genuine (0)", "Predicted Fraud (1)"],
        y=["Actual Genuine (0)", "Actual Fraud (1)"],
        text=[
            ["TN: 56,857<br>(99.99%)", "FP: 7<br>(0.01%)"],
            ["FN: 18<br>(18.37%)", "TP: 80<br>(81.63%)"]
        ],
        texttemplate="%{text}",
        textfont={"size": 14, "color": "white"},
        colorscale="Blues",
        showscale=False
    ))
    fig_cm.update_layout(
        xaxis=dict(tickfont=dict(size=12)),
        yaxis=dict(tickfont=dict(size=12), autorange="reversed"),
        margin=dict(l=40, r=40, t=20, b=40),
        height=320
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    c_tn, c_fp, c_fn, c_tp = st.columns(4)
    c_tn.metric("True Negatives (TN)", "56,857")
    c_fp.metric("False Positives (FP)", "7")
    c_fn.metric("False Negatives (FN)", "18")
    c_tp.metric("True Positives (TP)", "80")

with tab_batch:
    st.markdown('<div class="section-header">Batch Transaction Prediction</div>', unsafe_allow_html=True)
    st.markdown("Upload a CSV file containing transactions with the required 30 features (`Time`, `V1`–`V28`, `Amount`).")

    col_upload, col_sample_btn = st.columns([3, 1])
    with col_upload:
        uploaded_csv = st.file_uploader("Upload CSV File", type=["csv"], key="batch_uploader")
    with col_sample_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        use_sample_batch = st.button("📁 Try with Demo Batch (200 records)", use_container_width=True)

    batch_input_df = None
    if uploaded_csv is not None:
        try:
            batch_input_df = pd.read_csv(uploaded_csv)
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
    elif use_sample_batch:
        sample_batch_file = Path("sample_data/batch_test_sample.csv")
        if sample_batch_file.exists():
            batch_input_df = pd.read_csv(sample_batch_file)
        else:
            st.warning("Sample batch file not found.")

    if batch_input_df is not None:
        st.info(f"Loaded **{len(batch_input_df):,}** transactions for batch scoring.")
        try:
            annotated_df, summary = predict_batch(batch_input_df, artifacts)

            b1, b2, b3 = st.columns(3)
            b1.metric("Total Transactions", f"{summary['total_transactions']:,}")
            b2.metric("Flagged as Fraud", f"{summary['flagged_fraud_count']:,}")
            b3.metric("Fraud Rate", f"{summary['fraud_rate_pct']:.2f}%")

            display_cols = ["Prediction", "Fraud Probability (%)", "Amount", "Time", "V14", "V4", "V12", "V10"]
            existing_cols = [c for c in display_cols if c in annotated_df.columns]

            st.dataframe(
                annotated_df[existing_cols].style.apply(
                    lambda row: ['background-color: #FEF2F2' if row['Prediction'] == 'FRAUD' else '' for _ in row],
                    axis=1
                ).format({
                    "Amount": "${:,.2f}",
                    "Fraud Probability (%)": "{:.2f}%"
                }),
                use_container_width=True
            )

            csv_buffer = io.StringIO()
            annotated_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Scored Batch Results (CSV)",
                data=csv_buffer.getvalue(),
                file_name="credit_card_fraud_predictions.csv",
                mime="text/csv",
                type="primary"
            )

        except Exception as e:
            st.error(f"Batch prediction error: {e}")

with tab_info:
    st.markdown('<div class="section-header">Model Architecture & Specifications</div>', unsafe_allow_html=True)

    info_data = {
        "Property": [
            "Model Algorithm",
            "Problem Type",
            "Decision Threshold",
            "Input Features Count",
            "Imbalance Technique",
            "Feature Preprocessor",
            "Source Dataset",
            "Total Dataset Size"
        ],
        "Specification": [
            "XGBoost (Extreme Gradient Boosting)",
            "Binary Classification (0 = Genuine, 1 = Fraud)",
            f"{threshold:.2f}",
            "30 (Time, V1 – V28, Amount)",
            "SMOTE (Synthetic Minority Over-sampling Technique)",
            "StandardScaler",
            "creditcard.csv (European cardholders)",
            "284,807 transactions (492 fraud cases, 0.172%)"
        ]
    }
    st.table(pd.DataFrame(info_data))

with tab_explain:
    st.markdown('<div class="section-header">SHAP Feature Importance Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        "Based on SHAP (SHapley Additive exPlanations) analysis conducted during research, "
        "the following latent PCA components were identified as the highest-impact drivers for fraud detection:"
    )

    shap_importance_ranks = pd.DataFrame({
        "Rank": list(range(1, 11)),
        "Feature": TOP_SHAP_FEATURES,
        "Description": [
            "Anonymized PCA feature",
            "Anonymized PCA feature",
            "Anonymized PCA feature",
            "Anonymized PCA feature",
            "Anonymized PCA feature",
            "Anonymized PCA feature",
            "Anonymized PCA feature",
            "Anonymized PCA feature",
            "Anonymized PCA feature",
            "Anonymized PCA feature"
        ]
    })
    st.dataframe(shap_importance_ranks, use_container_width=True, hide_index=True)

    fig_shap = go.Figure(go.Bar(
        x=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        y=TOP_SHAP_FEATURES[::-1],
        orientation="h",
        marker=dict(color="#2563EB")
    ))
    fig_shap.update_layout(
        title="Top 10 High-Impact Features Identified by SHAP",
        xaxis_title="Relative Impact Rank",
        yaxis_title="Feature",
        height=340,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig_shap, use_container_width=True)
