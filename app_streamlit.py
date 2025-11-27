import streamlit as st
import pandas as pd
from pathlib import Path
import numpy as np

# Import joblib with a friendly error if it's missing
try:
    import joblib
except ModuleNotFoundError:
    st.error(
        "Missing dependency: 'joblib' is not installed in this environment. "
        "Install it by adding 'joblib' to requirements.txt or running 'pip install joblib'."
    )
    st.stop()

# Resolve model path relative to this file
MODEL_PATH = Path(__file__).resolve().parent / "models" / "credit_risk_pipeline.joblib"

st.title("Credit Risk - Demo")
st.write("Enter applicant info and get a predicted probability of 'bad credit'")

# Load model
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model from {MODEL_PATH}: {e}")
    st.stop()

# Numeric inputs
duration = st.number_input("Duration (months)", min_value=1, max_value=100, value=12, step=1)
amount = st.number_input("Credit Amount", min_value=0, value=0, step=1)
installment_rate = st.number_input("Installment Rate (%)", min_value=1, max_value=100, value=4, step=1)
present_residence = st.number_input("Residence Duration (years)", min_value=0, max_value=10, value=1, step=1)
age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
number_credits = st.number_input("Number Existing Credits", min_value=0, max_value=10, value=1, step=1)
people_liable = st.number_input("Number People Liable", min_value=0, max_value=10, value=1, step=1)

# Categorical inputs
status = st.selectbox("Checking Account Status", options=["A11", "A12", "A13", "A14"])
credit_history = st.selectbox("Credit History", options=["A30", "A31", "A32", "A33", "A34"])
purpose = st.selectbox("Purpose", options=["A40", "A41", "A42", "A43", "A44", "A45"])
personal_status_sex = st.selectbox("Personal Status & Sex", options=["A91", "A92", "A93", "A94"])
property = st.selectbox("Property", options=["A121", "A122", "A123", "A124"])
other_debtors = st.selectbox("Other Debtors", options=["A101", "A102", "A103"])
savings = st.selectbox("Savings", options=["A61", "A62", "A63", "A64", "A65"])
employment_duration = st.selectbox("Employment Duration", options=["A71", "A72", "A73", "A74", "A75"])
other_installment_plans = st.selectbox("Other Installment Plans", options=["A141", "A142", "A143"])
housing = st.selectbox("Housing", options=["A151", "A152", "A153"])
job = st.selectbox("Job", options=["A171", "A172", "A173", "A174"])
telephone = st.selectbox("Telephone", options=["A191", "A192"])

if st.button("Predict"):
    # Build DataFrame with all expected columns
    data = pd.DataFrame([{
        "duration": duration,
        "amount": amount,
        "installment_rate": installment_rate,
        "present_residence": present_residence,
        "age": age,
        "number_credits": number_credits,
        "people_liable": people_liable,
        "status": status,
        "credit_history": credit_history,
        "purpose": purpose,
        "personal_status_sex": personal_status_sex,
        "property": property,
        "other_debtors": other_debtors,
        "savings": savings,
        "employment_duration": employment_duration,
        "other_installment_plans": other_installment_plans,
        "housing": housing,
        "job": job,
        "telephone": telephone
    }])

    st.write("Input data being sent to model:")
    st.write(data)

    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(data)[:, 1][0]
        else:
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
