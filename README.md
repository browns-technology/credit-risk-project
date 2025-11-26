# credit-risk-project

Small demo repo containing a trained scikit-learn pipeline and two example apps:

- `app_fastapi.py` — FastAPI app exposing a `/predict` endpoint (uses `models/credit_risk_pipeline.joblib`).
- `app_streamlit.py` — Streamlit demo UI for entering sample applicant data and getting predictions.
- `load_model_with_shim.py` — Helper to attempt loading the joblib file and install a small shim if needed for older sklearn pickles.

Quick start (development):

1. Create a virtualenv and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run Streamlit demo:

```bash
streamlit run app_streamlit.py
```

3. Run FastAPI app locally:

```bash
uvicorn app_fastapi:app --host 0.0.0.0 --port 8000
```

If the model is not present in `models/credit_risk_pipeline.joblib`, the apps will fail to load the model. Use `load_model_with_shim.py` to help with debugging environment mismatches.
# credit-risk-project

Small demo repo containing a trained scikit-learn pipeline and two example apps:

- `app_fastapi.py` — FastAPI app exposing a `/predict` endpoint (uses `models/credit_risk_pipeline.joblib`).
- `app_streamlit.py` — Streamlit demo UI for entering sample applicant data and getting predictions.
- `load_model_with_shim.py` — Helper to attempt loading the joblib file and install a small shim if needed for older sklearn pickles.

Quick start (development):

1. Create a virtualenv and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run Streamlit demo:

```bash
streamlit run app_streamlit.py
```

3. Run FastAPI app locally:

```bash
uvicorn app_fastapi:app --host 0.0.0.0 --port 8000
```

If the model is not present in `models/credit_risk_pipeline.joblib`, the apps will fail to load the model. Use `load_model_with_shim.py` to help with debugging environment mismatches.
