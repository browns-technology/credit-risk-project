# app_fastapi.py
import joblib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

MODEL_PATH = "models/credit_risk_pipeline.joblib"
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Credit Risk Model API")

# Define expected request schema (use exact feature names from your dataframe)
class CreditRequest(BaseModel):
    # Example fields; ensure they match your dataset column names exactly
    Duration: int
    CreditAmount: float
    InstallmentRatePercentage: int
    ResidenceDuration: int
    Age: int
    NumberExistingCredits: int
    NumberPeopleMaintenance: int
    # categorical fields: supply string values that match training set categories
    CheckingAccountStatus: str
    CreditHistory: str
    Purpose: str
    # add the rest of categorical fields here...

@app.post("/predict")
def predict(req: CreditRequest):
    # convert to DataFrame with single row
    data = pd.DataFrame([req.dict()])
    proba = model.predict_proba(data)[:,1][0]
    pred = int(proba >= 0.5)
    return {"predicted_bad_credit": pred, "probability_bad": float(proba)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)