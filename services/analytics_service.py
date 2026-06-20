"""Read-only dashboard artifact access."""
from services.model_service import load_json
def dashboard_data():
    return {k:load_json(v) for k,v in {"metrics":"model_metrics.json","best":"best_model.json","importance":"feature_importance.json","matrices":"confusion_matrices.json","subgroups":"subgroup_evaluation.json","roc":"roc_curve_data.json","pr":"precision_recall_data.json"}.items()}
