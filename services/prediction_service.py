"""Stateless in-memory inference."""
import pandas as pd
from config import CLASS_MAPPING, FEATURES
from services.explanation_service import explain
from services.model_service import resolve_model
from services.validation_service import validate_input

def predict(data: dict) -> dict:
    values,requested=validate_input(data); key,model,entry=resolve_model(requested)
    frame=pd.DataFrame([[values[f] for f in FEATURES]],columns=FEATURES)
    predicted=int(model.predict(frame)[0]); raw=model.predict_proba(frame)[0]
    probs={int(c):float(p) for c,p in zip(model.classes_,raw)}
    observations,guidance=explain(values)
    return {"class_id":predicted,"category":CLASS_MAPPING[predicted],
      "wording":f"The entered indicators are most similar to the {CLASS_MAPPING[predicted].replace(' Pattern','')} category in the training dataset.",
      "confidence":round(max(probs.values())*100,1),"probabilities":{CLASS_MAPPING[i]:round(probs.get(i,0)*100,1) for i in range(3)},
      "model_key":key,"model_name":entry["display_name"],"observations":observations,"guidance":guidance}

