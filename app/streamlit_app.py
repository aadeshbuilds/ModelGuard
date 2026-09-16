import streamlit as st
import numpy as np
import pandas as pd
import sys
import os
import plotly.express as px

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.evaluator import Evaluator
from core.error_analysis import ErrorAnalyzer
from core.confidence import ConfidenceAnalyzer
from core.health_report import HealthReport

st.set_page_config(page_title="ModelGuard", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: radial-gradient(circle at 20% 0%, #1a1035 0%, #0b0d17 45%, #05060a 100%);
}

/* Hero title */
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 4.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #c4b5fd, #f9a8d4, #93c5fd, #c4b5fd);
    background-size: 300% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shine 5s linear infinite;
    margin-bottom: 0.3rem;
    letter-spacing: -0.02em;
    text-shadow: 0 0 60px rgba(167, 139, 250, 0.3);
}

.hero-title-row {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 0.3rem;
}
.hero-title-row .hero-title { margin-bottom: 0; }

@keyframes shine {
    to { background-position: 200% center; }
}
.hero-tagline {
    color: #9CA3AF;
    font-size: 1.05rem;
    margin-top: -0.3rem;
    margin-bottom: 2rem;
}

/* Hero section wrapper with glow orbs */
.hero-wrap {
    position: relative;
    padding: 2.5rem 0 1.5rem 0;
    overflow: visible;
}
.glow-orb {
    position: absolute;
    border-radius: 50%;
    filter: blur(80px);
    opacity: 0.55;
    z-index: -1;
}
.glow-orb-1 { width: 280px; height: 280px; background: #a78bfa; top: -60px; left: 0px; }
.glow-orb-2 { width: 220px; height: 220px; background: #f472b6; top: 20px; left: 260px; }
.glow-orb-3 { width: 200px; height: 200px; background: #60a5fa; top: -20px; left: 480px; }

.hero-pill {
    display: inline-block;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 30px;
    padding: 8px 18px;
    font-size: 0.95rem;
    color: #E5E7EB;
    font-weight: 500;
    margin-bottom: 2.2rem;
}
.hero-pill b { color: #c4b5fd; }

/* Glassmorphic metric cards */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.04);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 20px 22px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-6px) scale(1.02) perspective(600px) rotateX(3deg);
    box-shadow: 0 16px 40px rgba(167, 139, 250, 0.25);
    border-color: rgba(167, 139, 250, 0.5);
}
[data-testid="stMetricLabel"] { font-size: 0.8rem; opacity: 0.6; letter-spacing: 0.02em; }
[data-testid="stMetricValue"] { font-size: 1.9rem; font-weight: 700; }

/* Badges */
.badge {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    box-shadow: 0 0 16px currentColor;
}
.badge-green { background: rgba(74, 222, 128, 0.12); color: #4ADE80; }
.badge-yellow { background: rgba(250, 204, 21, 0.12); color: #FACC15; }
.badge-red { background: rgba(248, 113, 113, 0.12); color: #F87171; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    padding: 10px 18px;
    border: 1px solid transparent;
}
.stTabs [aria-selected="true"] {
    background: rgba(167,139,250,0.10);
    border-bottom: 2px solid #a78bfa;
}
.stTabs [data-baseweb="tab"]:hover {
    background: rgba(255,255,255,0.05);
    transition: background 0.2s ease;
    cursor: pointer;
}
.stTabs [data-baseweb="tab"] p {
    transition: color 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover p {
    color: #c4b5fd;
}

.stTabs [data-baseweb="tab"]:focus {
    outline: none !important;
    box-shadow: none !important;
}
.stTabs button {
    outline: none !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.02);
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.06);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-wrap">
    <div class="glow-orb glow-orb-1"></div>
    <div class="glow-orb glow-orb-2"></div>
    <div class="glow-orb glow-orb-3"></div>
        <div class="hero-title-row">
        <svg width="56" height="56" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="shieldGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#c4b5fd"/>
                    <stop offset="50%" stop-color="#f9a8d4"/>
                    <stop offset="100%" stop-color="#93c5fd"/>
                </linearGradient>
            </defs>
            <path d="M12 2L4 5V11C4 16.5 7.5 21.2 12 22C16.5 21.2 20 16.5 20 11V5L12 2Z" fill="url(#shieldGrad)" stroke="url(#shieldGrad)" stroke-width="0.5"/>
            <path d="M9.5 12L11 13.5L14.5 10" stroke="#0b0d17" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <span class="hero-title">ModelGuard</span>
    </div>
    <div class="hero-pill">Model reliability testing — <b>beyond accuracy.</b></div>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR: DATA SOURCE ----------------
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.divider()
    st.markdown("**Data Source**")
    data_source = st.radio("Choose input", ["Use demo models", "Upload your own predictions"], label_visibility="collapsed")

    if data_source == "Use demo models":
        model_choice = st.selectbox("Select a model", ["Logistic Regression", "Random Forest", "XGBoost"])
        model_file_map = {
            "Logistic Regression": "logreg",
            "Random Forest": "rf",
            "XGBoost": "xgb"
        }
        model_key = model_file_map[model_choice]
        predictions_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "predictions")

        y_test = np.load(os.path.join(predictions_dir, "y_test.npy"))
        y_pred = np.load(os.path.join(predictions_dir, f"y_pred_{model_key}.npy"))
        y_proba = np.load(os.path.join(predictions_dir, f"y_proba_{model_key}.npy"))
        data_loaded = True

    else:
        st.caption("Upload 3 CSVs: true labels, predicted labels, predicted probabilities (one column per class).")
        y_true_file = st.file_uploader("y_true.csv", type="csv")
        y_pred_file = st.file_uploader("y_pred.csv", type="csv")
        y_proba_file = st.file_uploader("y_proba.csv", type="csv")

        if y_true_file and y_pred_file and y_proba_file:
            y_test = pd.read_csv(y_true_file).values.ravel()
            y_pred = pd.read_csv(y_pred_file).values.ravel()
            y_proba = pd.read_csv(y_proba_file).values
            data_loaded = True
        else:
            st.warning("Upload all three files to continue.")
            data_loaded = False

    st.divider()
    st.caption("ModelGuard v1.0 · Classification reliability toolkit")

# ---------------- MAIN REPORT ----------------
if data_loaded:
    evaluator = Evaluator(y_test, y_pred)
    error_analyzer = ErrorAnalyzer(y_test, y_pred)
    confidence_analyzer = ConfidenceAnalyzer(y_test, y_pred, y_proba)
    class_labels = [f"Class {c}" for c in np.unique(y_test)]

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🔲 Confusion Matrix", "🔍 Error Analysis", "⚠️ Confidence Analysis", "🏥 Health Report"])
    # ---- TAB 1: OVERVIEW ----
    with tab1:
        st.subheader("Overall Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{evaluator.accuracy():.2%}")
        col2.metric("Macro F1", f"{evaluator.f1()['macro']:.2%}")
        col3.metric("Macro Recall", f"{evaluator.recall()['macro']:.2%}")

        cw_rate = confidence_analyzer.confidently_wrong_rate()
        col4.metric("Confidently Wrong Rate", f"{cw_rate:.2%}",
                     help="Of all wrong predictions, what % were made with >90% confidence")

        badge_class = "badge-green" if cw_rate < 0.10 else "badge-yellow" if cw_rate < 0.25 else "badge-red"
        badge_text = "Reliable" if cw_rate < 0.10 else "Caution" if cw_rate < 0.25 else "High Risk"
        st.markdown(f'<span class="badge {badge_class}">{badge_text}</span>', unsafe_allow_html=True)

        st.divider()
        st.caption("Weighted vs. Macro — why both matter")
        wcol1, wcol2 = st.columns(2)
        with wcol1:
            st.markdown("**Weighted Average** *(accounts for class frequency)*")
            st.write(f"Precision: {evaluator.precision()['weighted']:.2%}")
            st.write(f"Recall: {evaluator.recall()['weighted']:.2%}")
            st.write(f"F1: {evaluator.f1()['weighted']:.2%}")
        with wcol2:
            st.markdown("**Macro Average** *(treats all classes equally)*")
            st.write(f"Precision: {evaluator.precision()['macro']:.2%}")
            st.write(f"Recall: {evaluator.recall()['macro']:.2%}")
            st.write(f"F1: {evaluator.f1()['macro']:.2%}")

    # ---- TAB 2: CONFUSION MATRIX ----
    with tab2:
        st.subheader("Confusion Matrix")
        cm = evaluator.confusion()
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="True", color="Count"),
            x=class_labels,
            y=class_labels,
            text_auto=True,
            color_continuous_scale="Blues"
        )
        fig.update_layout(height=550, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # ---- TAB 3: ERROR ANALYSIS ----
    with tab3:
        st.subheader("Worst-Performing Classes")
        error_rates = error_analyzer.error_rate_by_class()
        confused_with = error_analyzer.most_confused_with()

        error_summary = pd.DataFrame({
            'Class': error_rates.index,
            'Error Rate': error_rates.values,
            'Most Confused With': [confused_with.get(cls, '-') for cls in error_rates.index]
        })
        st.dataframe(
            error_summary.style.format({'Error Rate': '{:.2%}'}).background_gradient(subset=['Error Rate'], cmap='Reds'),
            use_container_width=True,
            hide_index=True
        )

        st.divider()
        st.subheader("Misclassified Examples")
        st.dataframe(error_analyzer.misclassified(), use_container_width=True, hide_index=True)

    # ---- TAB 4: CONFIDENCE ANALYSIS ----
    with tab4:
        st.subheader("Confidently Wrong Predictions")
        st.caption("Predictions where the model was highly confident but still incorrect.")
        threshold = st.slider("Confidence threshold", 0.5, 0.99, 0.9, 0.01)
        cw_df = confidence_analyzer.confidently_wrong(threshold)
        st.write(f"**{len(cw_df)}** confidently wrong predictions at this threshold.")
        st.dataframe(
            cw_df.style.background_gradient(subset=['confidence'], cmap='Oranges'),
            use_container_width=True,
            hide_index=True
        )

    # ---- TAB 5: HEALTH REPORT ----
    with tab5:
        st.subheader("Model Health Report")
        st.caption("Automated pass/warning/fail checks with suggestions.")

        health = HealthReport(y_test, y_pred, y_proba)
        checks = health.generate()

        status_styles = {
            "✅ Pass": ("rgba(74, 222, 128, 0.08)", "#4ADE80"),
            "⚠️ Warning": ("rgba(250, 204, 21, 0.08)", "#FACC15"),
            "❌ Fail": ("rgba(248, 113, 113, 0.08)", "#F87171"),
        }

        for check in checks:
            bg_color, border_color = status_styles[check["Status"]]
            st.markdown(f"""
            <div style="
                background: {bg_color};
                border-left: 4px solid {border_color};
                border-radius: 8px;
                padding: 14px 18px;
                margin-bottom: 12px;
            ">
                <div style="font-weight: 700; font-size: 1rem;">{check['Status']} &nbsp; {check['Check']}</div>
                <div style="opacity: 0.85; font-size: 0.9rem; margin-top: 4px;">{check['Details']}</div>
            </div>
            """, unsafe_allow_html=True)

else:
    st.info("👈 Choose a data source in the sidebar to get started.")