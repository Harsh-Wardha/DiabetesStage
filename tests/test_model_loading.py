from services.model_service import load_json, resolve_model

def test_best_reads_artifact():
    expected=load_json("best_model.json")["model_key"]
    key,model,entry=resolve_model("best")
    assert key == expected

def test_feature_order():
    from config import FEATURES
    _,_,entry=resolve_model("best")
    assert entry["feature_order"] == FEATURES
