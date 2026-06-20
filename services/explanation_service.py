"""General rule-based educational observations; never diagnostic."""
def explain(values: dict) -> tuple[list[str],list[str]]:
    observations=[]
    rules=[("HighBP","High blood pressure was reported among the entered indicators."),("HighChol","High cholesterol was reported among the entered indicators."),
      ("Smoker","A lifetime smoking indicator was reported."),("DiffWalk","Serious difficulty walking or climbing stairs was reported."),
      ("Stroke","A history-of-stroke indicator was reported."),("HeartDiseaseorAttack","A heart disease or heart attack indicator was reported.")]
    observations += [text for key,text in rules if values[key]==1]
    if values["BMI"]>=25: observations.append("The entered BMI is above the commonly referenced adult range; BMI alone does not determine health status.")
    if values["GenHlth"]>=4: observations.append("Overall health was described as fair or poor in the entered survey indicators.")
    if values["PhysActivity"]==0: observations.append("No leisure-time physical activity in the past 30 days was reported.")
    if not observations: observations.append("No specific rule-based observations were highlighted; the model still considered all entered indicators.")
    guidance=["Consider discussing personal risk factors and appropriate screening with a qualified healthcare professional.",
      "Seek professional advice about sustainable nutrition, activity, sleep and tobacco-use choices appropriate for you.",
      "If you have urgent or concerning symptoms, contact a healthcare professional or local emergency service."]
    return observations,guidance

