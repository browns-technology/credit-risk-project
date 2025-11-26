# app_streamlit.py
import streamlit as st
import pandas as pd
from pathlib import Path

# Import joblib with a friendly error if it's missing
try:
    import joblib
except ModuleNotFoundError:
    st.error(
        "Missing dependency: 'joblib' is not installed in this environment. "
        "Install it by adding 'joblib' to requirements.txt or running 'pip install joblib'."
    )
    st.stop()

# Resolve model path relative to this file to avoid FileNotFoundError when working dir differs
MODEL_PATH = Path(__file__).resolve().parent / "models" / "credit_risk_pipeline.joblib"

# app_streamlit.py
import streamlit as st
import pandas as pd
from pathlib import Path

# Import joblib with a friendly error if it's missing
try:
    import joblib
except ModuleNotFoundError:
    st.error(
        "Missing dependency: 'joblib' is not installed in this environment. "
        "Install it by adding 'joblib' to requirements.txt or running 'pip install joblib'."
    )
    st.stop()

# Resolve model path relative to this file to avoid FileNotFoundError when working dir differs
MODEL_PATH = Path(__file__).resolve().parent / "models" / "credit_risk_pipeline.joblib"

st.title("Credit Risk - Demo")
st.write("Enter applicant info and get a predicted probability of 'bad credit'")

# Try to load model with error handling
model = None
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model from {MODEL_PATH}: {e}")
    st.stop()

# numeric inputs (example; ensure names match dataset)
Duration = st.number_input("Duration (months)", min_value=1, max_value=100, value=12, step=1)
CreditAmount = st.number_input("Credit Amount", min_value=0, value=0, step=1)
InstallmentRatePercentage = st.number_input("Installment Rate (%)", min_value=1, max_value=100, value=4, step=1)
ResidenceDuration = st.number_input("Residence Duration (years)", min_value=0, max_value=10, value=1, step=1)
Age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
NumberExistingCredits = st.number_input("Number Existing Credits", min_value=0, max_value=10, value=1, step=1)
NumberPeopleMaintenance = st.number_input("Number People Maintenance", min_value=0, max_value=10, value=1, step=1)

# sample categorical fields - adapt to your dataset
CheckingAccountStatus = st.selectbox("Checking Account Status", options=["A11", "A12", "A13", "A14"])
CreditHistory = st.selectbox("Credit History", options=["A30", "A31", "A32", "A33", "A34"])
Purpose = st.selectbox("Purpose", options=["A40", "A41", "A42", "A43", "A44", "A45"])

if st.button("Predict"):
    data = pd.DataFrame([{
        "Duration": Duration,
        "CreditAmount": CreditAmount,
        "InstallmentRatePercentage": InstallmentRatePercentage,
        "ResidenceDuration": ResidenceDuration,
        "Age": Age,
        "NumberExistingCredits": NumberExistingCredits,
        "NumberPeopleMaintenance": NumberPeopleMaintenance,
        "CheckingAccountStatus": CheckingAccountStatus,
        "CreditHistory": CreditHistory,
        "Purpose": Purpose
    }])

    st.write("Input data being sent to model:")
    st.write(data)

    # Safely call the model and handle common errors (missing columns, missing method, etc.)
    try:
        if hasattr(model, "predict_proba"):
            # If model is a pipeline, ensure columns match; predict_proba returns array-like
            proba = model.predict_proba(data)[:, 1][0]
        else:
            # fallback if no predict_proba; show predicted class instead
            pred = model.predict(data)[0]
            st.warning("Model does not implement predict_proba; returning predicted class instead of probability.")
            proba = float(pred)

        st.metric("Probability bad credit", f"{proba:.3f}")
        st.write("Action suggestion:")
        if proba > 0.6:
            st.error("High risk — reject or request collateral / higher interest")
        elif proba > 0.3:
            st.warning("Medium risk — consider manual review")
        else:
            st.success("Low risk — proceed")
    except Exception as e:
        st.exception(e)
        st.error("Prediction failed. Check the input data above and confirm the model's expected feature names and categories.")
