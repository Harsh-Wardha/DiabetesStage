"""Server-side prediction input validation."""
from config import BINARY_FEATURES, FEATURES

RANGES={"BMI":(10,100),"GenHlth":(1,5),"MentHlth":(0,30),"PhysHlth":(0,30),"Age":(1,13)}
MODELS={"best","logistic_regression","decision_tree","hist_gradient_boosting"}

class ValidationError(ValueError): pass

def validate_input(data: dict) -> tuple[dict[str,float],str]:
    missing=[f for f in FEATURES if data.get(f) in (None,"")]
    if missing: raise ValidationError("Please complete all fields: "+", ".join(missing))
    clean={}
    for field in FEATURES:
        try: value=float(data[field])
        except (TypeError,ValueError): raise ValidationError(f"{field} must be numeric.")
        if field in BINARY_FEATURES and value not in (0,1): raise ValidationError(f"{field} must be 0 or 1.")
        if field in RANGES and not RANGES[field][0] <= value <= RANGES[field][1]: raise ValidationError(f"{field} must be between {RANGES[field][0]} and {RANGES[field][1]}.")
        if field != "BMI" and not value.is_integer(): raise ValidationError(f"{field} must be a whole number.")
        clean[field]=value
    model=str(data.get("model","best"))
    if model not in MODELS: raise ValidationError("Unknown model selection.")
    return clean,model

