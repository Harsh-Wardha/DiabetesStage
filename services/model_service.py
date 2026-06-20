"""Read-only, lazy model and artifact loading."""
from functools import lru_cache
import json, logging
from pathlib import Path
import joblib
from config import BASE_DIR, FEATURES

log=logging.getLogger(__name__)
UNAVAILABLE="Prediction models are unavailable. Run python train_models.py locally and commit the generated model artifacts before deploying."

class ModelUnavailableError(RuntimeError): pass

@lru_cache(maxsize=None)
def load_json(name: str):
    path=BASE_DIR/"artifacts"/name
    try: return json.loads(path.read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError) as exc: log.exception("Artifact load failure: %s",name); raise ModelUnavailableError(UNAVAILABLE) from exc

@lru_cache(maxsize=3)
def load_model(key: str):
    registry_path=BASE_DIR/"models/model_registry.json"
    try:
        registry=json.loads(registry_path.read_text(encoding="utf-8")); entry=registry[key]
        if entry.get("feature_order") != FEATURES: raise ValueError("Feature order mismatch")
        path=BASE_DIR/"models"/entry["artifact_filename"]
        if not path.is_file(): raise FileNotFoundError(path)
        return joblib.load(path),entry
    except Exception as exc: log.exception("Model load failure: %s",key); raise ModelUnavailableError(UNAVAILABLE) from exc

def resolve_model(key: str):
    if key=="best": key=load_json("best_model.json")["model_key"]
    return key,*load_model(key)

