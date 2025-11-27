# Model artifacts

The trained pipeline (if present) is stored at:

`models/credit_risk_pipeline.joblib`

This artifact is a joblib/pickle file created with scikit-learn. Unpickling can fail across scikit-learn versions because private/internal classes and names change between releases (for example: `_RemainderColsList` in `sklearn.compose._column_transformer`). If you see errors like:

    Can't get attribute '_RemainderColsList' on <module 'sklearn.compose._column_transformer' ...>

it means the environment you are using to load the model does not match the scikit-learn version used to create it.

Recommended steps

1. Try to detect the scikit-learn version used to create the model (best-effort):

```bash
# raw inspection for strings mentioning scikit or scikit-learn
strings models/credit_risk_pipeline.joblib | sed -n '1,200p' | grep -i scikit || true
# or a small Python peek that reads the first bytes and searches for version-like strings
python3 - <<'PY'
p = 'models/credit_risk_pipeline.joblib'
s = open(p, 'rb').read(20000)
print(s.decode('latin1', errors='ignore')[:2000])
PY
```

2. If you can find the sklearn version, create an environment pinned to that version and load the model there (recommended for reliability):

```bash
python3 -m venv venv-sklearn-old
source venv-sklearn-old/bin/activate
pip install "scikit-learn==<VERSION>" joblib
python3 -c "import joblib; joblib.load('models/credit_risk_pipeline.joblib')"
```

3. Quick temporary workaround (shim): the repository contains `load_model_with_shim.py` which installs a minimal shim for `_RemainderColsList` and tries to load the joblib. This is a debugging/inspection helper and not a guaranteed production fix.

4. ONNX conversion (optional): there is no conversion script included by default in this repo. If you want to convert the pipeline to ONNX, install `skl2onnx` and `onnx` and either add a small converter script or perform the conversion interactively. Conversion requires knowledge of model input shape and feature names.

Example conversion (manual):

```bash
# install converter
pip install skl2onnx onnx

# then in Python, load model and convert (simplified example):
python3 - <<'PY'
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

model = joblib.load('models/credit_risk_pipeline.joblib')
initial_type = [("input", FloatTensorType([None, <N_FEATURES>]))]
onx = convert_sklearn(model, initial_types=initial_type)
with open('models/credit_risk_pipeline.onnx', 'wb') as f:
    f.write(onx.SerializeToString())
PY
```

Notes

- If conversion fails because the pipeline contains custom or unsupported transformers, load the model in the original sklearn environment and re-create a simplified pipeline that only contains components supported by `skl2onnx`, or export model parameters and reconstruct.
# Model artifacts

The trained pipeline (if present) is stored at:

`models/credit_risk_pipeline.joblib`

This artifact is a joblib/pickle file created with scikit-learn. Unpickling can fail across scikit-learn versions because private/internal classes and names change between releases (for example: `_RemainderColsList` in `sklearn.compose._column_transformer`). If you see errors like:

    Can't get attribute '_RemainderColsList' on <module 'sklearn.compose._column_transformer' ...>

it means the environment you are using to load the model does not match the scikit-learn version used to create it.

Recommended steps

1. Try to detect the scikit-learn version used to create the model (best-effort):

```bash
# raw inspection for strings mentioning scikit or scikit-learn
strings models/credit_risk_pipeline.joblib | sed -n '1,200p' | grep -i scikit || true
# or a small Python peek that reads the first bytes and searches for version-like strings
python3 - <<'PY'
p = 'models/credit_risk_pipeline.joblib'
s = open(p, 'rb').read(20000)
print(s.decode('latin1', errors='ignore')[:2000])
PY
```

2. If you can find the sklearn version, create an environment pinned to that version and load the model there (recommended for reliability):

```bash
python3 -m venv venv-sklearn-old
source venv-sklearn-old/bin/activate
pip install "scikit-learn==<VERSION>" joblib
python3 -c "import joblib; joblib.load('models/credit_risk_pipeline.joblib')"
```

3. Quick temporary workaround (shim): the repository contains `load_model_with_shim.py` which installs a minimal shim for `_RemainderColsList` and tries to load the joblib. This is a debugging/inspection helper and not a guaranteed production fix.

4. ONNX conversion (optional): there is no conversion script included by default in this repo. If you want to convert the pipeline to ONNX, install `skl2onnx` and `onnx` and either add a small converter script or perform the conversion interactively. Conversion requires knowledge of model input shape and feature names.

Example conversion (manual):

```bash
# install converter
pip install skl2onnx onnx

# then in Python, load model and convert (simplified example):
python3 - <<'PY'
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

model = joblib.load('models/credit_risk_pipeline.joblib')
initial_type = [("input", FloatTensorType([None, <N_FEATURES>]))]
onx = convert_sklearn(model, initial_types=initial_type)
with open('models/credit_risk_pipeline.onnx', 'wb') as f:
    f.write(onx.SerializeToString())
PY
```

Notes

- If conversion fails because the pipeline contains custom or unsupported transformers, load the model in the original sklearn environment and re-create a simplified pipeline that only contains components supported by `skl2onnx`, or export model parameters and reconstruct.
