import streamlit as st
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.evaluator import Evaluator
from core.error_analysis import ErrorAnalyzer
from core.confidence import ConfidenceAnalyzer

st.set_page_config(page_title="ModelGuard", layout="wide")
st.title("🛡️ ModelGuard")
st.write("Model reliability testing — beyond accuracy.")

st.header("Choose Data Source")
data_source = st.radio("Data source", ["Use demo models", "Upload your own predictions"])

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
    st.write("Upload three CSV files: true labels, predicted labels, and predicted probabilities (one column per class).")
    y_true_file = st.file_uploader("y_true.csv (single column)", type="csv")
    y_pred_file = st.file_uploader("y_pred.csv (single column)", type="csv")
    y_proba_file = st.file_uploader("y_proba.csv (one column per class)", type="csv")

    if y_true_file and y_pred_file and y_proba_file:
        y_test = pd.read_csv(y_true_file).values.ravel()
        y_pred = pd.read_csv(y_pred_file).values.ravel()
        y_proba = pd.read_csv(y_proba_file).values
        data_loaded = True
    else:
        st.warning("Please upload all three files to continue.")
        data_loaded = False

if data_loaded:
    evaluator = Evaluator(y_test, y_pred)
    error_analyzer = ErrorAnalyzer(y_test, y_pred)
    confidence_analyzer = ConfidenceAnalyzer(y_test, y_pred, y_proba)

    st.header("Overall Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{evaluator.accuracy():.2%}")
    col2.metric("Macro F1", f"{evaluator.f1()['macro']:.2%}")
    col3.metric("Macro Recall", f"{evaluator.recall()['macro']:.2%}")
    col4.metric("Confidently Wrong Rate", f"{confidence_analyzer.confidently_wrong_rate():.2%}")