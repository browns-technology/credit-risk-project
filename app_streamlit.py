# app_streamlit.py
import joblib
import streamlit as st
import pandas as pd

model = joblib.load("models/credit_risk_pipeline.joblib")

st.title("Credit Risk - Demo")
st.write("Enter applicant info and get a predicted probability of 'bad credit'")

# numeric inputs (example; ensure names match dataset)
Duration = st.number_input("Duration (months)", min_value=1, max_value=100, value=12)
CreditAmount = st.number_input("Credit Amount", min_value=0)
InstallmentRatePercentage = st.number_input("Installment Rate (%)", min_value=1, max_value=100, value=4)
ResidenceDuration = st.number_input("Residence Duration (years)", min_value=0, max_value=10, value=1)
Age = st.number_input("Age", min_value=18, max_value=100, value=35)
NumberExistingCredits = st.number_input("Number Existing Credits", min_value=0, max_value=10, value=1)
NumberPeopleMaintenance = st.number_input("Number People Maintenance", min_value=0, max_value=10, value=1)

# sample categorical fields - adapt to your dataset
# Replace the options below with the categories used in your training data
CheckingAccountStatus = st.selectbox("Checking Account Status", options=["A11","A12","A13","A14"])
CreditHistory = st.selectbox("Credit History", options=["A30","A31","A32","A33","A34"])
Purpose = st.selectbox("Purpose", options=["A40","A41","A42","A43","A44","A45"])

if st.button("Predict"):
    data = pd.DataFrame([{
        "Duration": Duration,
        "CreditAmount": CreditAmount,
        "InstallmentRatePercentage": InstallmentRatePercentage,
        "ResidenceDuration": ResidenceDuration,
        "Age": Age,
        "NumberExistingCredits": NumberExistingCredits,
        "NumberPeopleMaintenance": NumberPeopleMaintenance,
        # add categorical fields (names must match training data)
        "CheckingAccountStatus": CheckingAccountStatus,
        "CreditHistory": CreditHistory,
        "Purpose": Purpose
    }])
    proba = model.predict_proba(data)[:,1][0]
    st.metric("Probability bad credit", f"{proba:.3f}")
    st.write("Action suggestion:")
    if proba > 0.6:
        st.error("High risk — reject or request collateral / higher interest")
    elif proba > 0.3:
        st.warning("Medium risk — consider manual review")
    else:
        st.success("Low risk — proceed")