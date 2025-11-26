# Model artifacts

The trained pipeline is stored at:

/mount/src/credit-risk-project/models/credit_risk_pipeline.joblib

This artifact is a joblib/pickle file created with scikit-learn. Unpickling can fail across scikit-learn versions because private/internal classes and names change between releases (for example: `_RemainderColsList` in `sklearn.compose._column_transformer`). If you see errors like:

    Can't get attribute '_RemainderColsList' on <module 'sklearn.compose._column_transformer' ...>

it means the environment you are using to load the model does not match the scikit-learn version used to create it.

Recommended steps

1. Try to detect the scikit-learn version used to create the model (best-effort):

```bash
# raw inspection for strings mentioning scikit or scikit-learn
strings /mount/src/credit-risk-project/models/credit_risk_pipeline.joblib | sed -n '1,200p' | grep -i scikit || true
# or a small Python peek that reads the first bytes and searches for version-like strings
python3 - <<'PY'
p = '/mount/src/credit-risk-project/models/credit_risk_pipeline.joblib'
s = open(p, 'rb').read(20000)
print(s.decode('latin1', errors='ignore')[:2000])
PY
```

2. If you can find the sklearn version, create an environment pinned to that version and load the model there (recommended for reliability):

```bash
python3 -m venv venv-sklearn-old
source venv-sklearn-old/bin/activate
pip install "scikit-learn==<VERSION>" joblib
python3 -c "import joblib; joblib.load('/mount/src/credit-risk-project/models/credit_risk_pipeline.joblib')"
```

3. Quick temporary workaround (shim): the repository contains `load_model_with_shim.py` which installs a minimal shim for `_RemainderColsList` and tries to load the joblib. This is a debugging/inspection helper and not a guaranteed production fix.

4. Conversion to ONNX (portable): the repository now includes `scripts/convert_to_onnx.py`. Use it to attempt to convert the scikit-learn pipeline to ONNX. Conversion requires knowledge of the model input shape / feature names; the converter supports using `feature_names_in_` if present, a CSV sample row, or an explicit `--n-features` fallback.

Example usage:

```bash
# Try a default conversion (writes to same folder with .onnx suffix)
python3 scripts/convert_to_onnx.py

# Provide a sample CSV with one row to infer feature names and types
python3 scripts/convert_to_onnx.py --sample-csv data/sample_row.csv --output /mount/src/credit-risk-project/models/credit_risk_pipeline.onnx

# Provide a fallback number of features (all floats)
python3 scripts/convert_to_onnx.py --n-features 20 --output /mount/src/credit-risk-project/models/credit_risk_pipeline.onnx
```

Notes

- The converter uses `skl2onnx`. Install it in your environment before running the converter: `pip install skl2onnx onnx`.
- If conversion fails due to complex or custom transformers, load the model in the original sklearn environment and re-create a simplified pipeline that only contains components supported by `skl2onnx`, or export model parameters and reconstruct.
