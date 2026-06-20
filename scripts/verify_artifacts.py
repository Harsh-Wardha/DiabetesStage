"""Verify committed inference artifacts end to end."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, joblib
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from config import FEATURES
REQUIRED=["best_model.json","model_metrics.json","training_metadata.json","data_preparation_summary.json","classification_reports.json","confusion_matrices.json","roc_curve_data.json","precision_recall_data.json","feature_importance.json","dataset_insights.json","subgroup_evaluation.json","feature_metadata.json"]
def main():
    errors=[]
    for name in REQUIRED:
        try: json.loads((ROOT/"artifacts"/name).read_text(encoding="utf-8"))
        except Exception as e: errors.append(f"{name}: {e}")
    try: registry=json.loads((ROOT/"models/model_registry.json").read_text(encoding="utf-8")); best=json.loads((ROOT/"artifacts/best_model.json").read_text(encoding="utf-8"))["model_key"]
    except Exception as e: registry={}; best=None; errors.append(f"Registry/best model: {e}")
    sample=pd.DataFrame([[1,1,1,28,0,0,0,1,1,1,0,3,2,2,0,7]],columns=FEATURES)
    for key,entry in registry.items():
        try:
            if entry["feature_order"]!=FEATURES: raise ValueError("feature order mismatch")
            model=joblib.load(ROOT/"models"/entry["artifact_filename"]); pred=int(model.predict(sample)[0]); p=model.predict_proba(sample)[0]
            if sorted(model.classes_.astype(int).tolist()) != [0,1,2]: raise ValueError("model does not recognize all classes")
            if pred not in [0,1,2] or not np.isclose(p.sum(),1): raise ValueError("invalid prediction probabilities")
            print(f"{key}: {(ROOT/'models'/entry['artifact_filename']).stat().st_size:,} bytes; sample class {pred}")
        except Exception as e: errors.append(f"{key}: {e}")
    if best not in registry: errors.append("Best-model artifact does not identify a registered model")
    if errors: print("Verification failed:\n"+"\n".join(errors),file=sys.stderr); raise SystemExit(1)
    print("All model and JSON artifacts verified successfully.")
if __name__=="__main__": main()

