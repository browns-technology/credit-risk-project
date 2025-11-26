#!/usr/bin/env python3
import importlib
import joblib
import sys
from pathlib import Path

MODEL_PATH = "/mount/src/credit-risk-project/models/credit_risk_pipeline.joblib"

def ensure_remainder_shim():
    try:
        mod = importlib.import_module("sklearn.compose._column_transformer")
    except Exception as e:
        print("Could not import sklearn.compose._column_transformer:", e, file=sys.stderr)
        return False

    if not hasattr(mod, "_RemainderColsList"):
        # Minimal shim: a simple subclass of list so unpickling can find the name.
        class _RemainderColsList(list):
            pass
        setattr(mod, "_RemainderColsList", _RemainderColsList)
        print("Shim installed: sklearn.compose._column_transformer._RemainderColsList")
    else:
        print("Shim already present.")
    return True

def load_model(path):
    try:
        return joblib.load(path)
    except AttributeError as e:
        print("AttributeError on first load attempt:", e)
        print("Attempting to install shim and retry...")
        ensure_remainder_shim()
        return joblib.load(path)

if __name__ == "__main__":
    p = Path(MODEL_PATH)
    if not p.exists():
        print(f"Model not found at {MODEL_PATH}", file=sys.stderr)
        sys.exit(2)
    try:
        model = load_model(str(p))
        print("Model loaded. Type:", type(model))
    except Exception as e:
        print("Failed to load model:", repr(e), file=sys.stderr)
        sys.exit(3)