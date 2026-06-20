"""Train, evaluate, select and persist all DiabetesStage models."""
from __future__ import annotations
import json, platform, sys, time
from datetime import datetime, timezone
from pathlib import Path
import joblib, numpy as np, pandas as pd, sklearn
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
 classification_report, confusion_matrix, f1_score, precision_recall_curve, precision_score,
 recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from config import BASE_DIR, CLASS_MAPPING, FEATURES, TARGET

DATA=BASE_DIR/"dataset/processed/diabetes_stage_clean.csv"; MODELS=BASE_DIR/"models"; ART=BASE_DIR/"artifacts"; REPORTS=BASE_DIR/"reports"
KEYS=["logistic_regression","decision_tree","hist_gradient_boosting"]
NAMES={"logistic_regression":"Logistic Regression","decision_tree":"Decision Tree","hist_gradient_boosting":"Histogram Gradient Boosting"}

def dump(path: Path, value) -> None: path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
def native(v):
    if isinstance(v, dict): return {str(k):native(x) for k,x in v.items()}
    if isinstance(v, (list,tuple,np.ndarray)): return [native(x) for x in v]
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return None if not np.isfinite(v) else float(v)
    if isinstance(v, float): return None if not np.isfinite(v) else v
    return v

def evaluate(model, x, y, train_seconds):
    start=time.perf_counter(); pred=model.predict(x); proba=model.predict_proba(x); inference=(time.perf_counter()-start)/len(x)*1000
    report=classification_report(y,pred,labels=[0,1,2],target_names=[CLASS_MAPPING[i] for i in range(3)],output_dict=True,zero_division=0)
    metrics={"accuracy":accuracy_score(y,pred),"balanced_accuracy":balanced_accuracy_score(y,pred),
      "macro_precision":precision_score(y,pred,average="macro",zero_division=0),"macro_recall":recall_score(y,pred,average="macro",zero_division=0),
      "macro_f1":f1_score(y,pred,average="macro",zero_division=0),"weighted_precision":precision_score(y,pred,average="weighted",zero_division=0),
      "weighted_recall":recall_score(y,pred,average="weighted",zero_division=0),"weighted_f1":f1_score(y,pred,average="weighted",zero_division=0),
      "training_time_seconds":train_seconds,"average_prediction_time_ms":inference}
    try: metrics["roc_auc_ovr_macro"]=roc_auc_score(y,proba,multi_class="ovr",average="macro")
    except ValueError: metrics["roc_auc_ovr_macro"]=None
    cm=confusion_matrix(y,pred,labels=[0,1,2]); cmn=confusion_matrix(y,pred,labels=[0,1,2],normalize="true")
    curves={}; prs={}; binary=label_binarize(y,classes=[0,1,2])
    for i in range(3):
        fpr,tpr,thr=roc_curve(binary[:,i],proba[:,i]); p,r,t=precision_recall_curve(binary[:,i],proba[:,i])
        # Preserve curve shape while keeping production JSON compact.
        ri=np.unique(np.linspace(0,len(fpr)-1,min(500,len(fpr)),dtype=int)); pi=np.unique(np.linspace(0,len(p)-1,min(500,len(p)),dtype=int))
        ti=ri[ri < len(thr)]; pti=pi[pi < len(t)]
        curves[str(i)]={"fpr":fpr[ri].tolist(),"tpr":tpr[ri].tolist(),"thresholds":thr[ti].tolist()}
        prs[str(i)]={"precision":p[pi].tolist(),"recall":r[pi].tolist(),"thresholds":t[pti].tolist(),"average_precision":average_precision_score(binary[:,i],proba[:,i])}
    return native(metrics),native(report),native({"raw":cm,"normalized":cmn}),native(curves),native(prs),pred

def insights(df, summary):
    counts=df[TARGET].value_counts().sort_index(); total=len(df)
    hist,edges=np.histogram(df.BMI,bins=[10,18.5,25,30,35,40,50,60,75,100])
    return {"total_cleaned_records":total,"feature_count":len(FEATURES),"class_counts":{str(k):int(v) for k,v in counts.items()},
      "class_percentages":{str(k):round(v/total*100,2) for k,v in counts.items()},"duplicate_rows_removed":summary.get("duplicate_rows_removed",0),
      "invalid_rows_removed":summary.get("invalid_rows_removed",0),"bmi_summary":native(df.BMI.describe().to_dict()),
      "age_distribution":{str(k):int(v) for k,v in df.Age.value_counts().sort_index().items()},
      "general_health_distribution":{str(k):int(v) for k,v in df.GenHlth.value_counts().sort_index().items()},
      "bmi_histogram":{"labels":[f"{edges[i]:g}–{edges[i+1]:g}" for i in range(len(edges)-1)],"counts":hist.tolist()},
      "general_health_by_target":{str(t):{str(k):int(v) for k,v in g.GenHlth.value_counts().sort_index().items()} for t,g in df.groupby(TARGET)},
      "indicator_percentages":{c:round(float(df[c].mean()*100),2) for c in ["HighBP","HighChol","PhysActivity","Smoker","DiffWalk"]},
      "by_target":{c:{str(t):round(float(g[c].mean()*100),2) for t,g in df.groupby(TARGET)} for c in ["HighBP","HighChol","PhysActivity","DiffWalk"]}}

def main():
    if not DATA.exists(): raise SystemExit("Cleaned dataset missing. Run python scripts/prepare_dataset.py first.")
    MODELS.mkdir(exist_ok=True); ART.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)
    df=pd.read_csv(DATA); classes=sorted(df[TARGET].astype(int).unique().tolist())
    if classes != [0,1,2]: raise SystemExit(f"Three-class target required; found {classes}")
    x=df[FEATURES]; y=df[TARGET].astype(int)
    indices=np.arange(len(df)); train_i,test_i=train_test_split(indices,test_size=.2,random_state=42,stratify=y)
    xtrain,xtest=x.iloc[train_i],x.iloc[test_i]; ytrain,ytest=y.iloc[train_i],y.iloc[test_i]
    models={
      "logistic_regression":Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler()),("classifier",LogisticRegression(max_iter=3000,class_weight="balanced",solver="lbfgs",random_state=42))]),
      "decision_tree":Pipeline([("imputer",SimpleImputer(strategy="median")),("classifier",DecisionTreeClassifier(max_depth=10,min_samples_split=30,min_samples_leaf=15,class_weight="balanced",random_state=42))]),
      "hist_gradient_boosting":Pipeline([("imputer",SimpleImputer(strategy="median")),("classifier",HistGradientBoostingClassifier(max_iter=150,learning_rate=.08,max_leaf_nodes=31,l2_regularization=.1,random_state=42))])}
    all_metrics={}; reports={}; matrices={}; rocs={}; prs={}; predictions={}; importance={}; registry={}; timestamp=datetime.now(timezone.utc).isoformat()
    for key,model in models.items():
        print(f"Training {NAMES[key]}..."); started=time.perf_counter()
        if key=="hist_gradient_boosting": model.fit(xtrain,ytrain,classifier__sample_weight=compute_sample_weight(class_weight="balanced",y=ytrain))
        else: model.fit(xtrain,ytrain)
        elapsed=time.perf_counter()-started
        metrics,report,cm,roc,pr,pred=evaluate(model,xtest,ytest,elapsed)
        artifact=MODELS/f"{key}.joblib"; joblib.dump(model,artifact,compress=3); size=artifact.stat().st_size; metrics["artifact_size_bytes"]=size
        clf=model.named_steps["classifier"]
        if key=="logistic_regression": vals=np.abs(clf.coef_).mean(axis=0)
        elif key=="decision_tree": vals=clf.feature_importances_
        else:
            rng=np.random.default_rng(42); sample=np.sort(rng.choice(len(xtest),min(4000,len(xtest)),replace=False))
            vals=permutation_importance(model,xtest.iloc[sample],ytest.iloc[sample],scoring="f1_macro",n_repeats=3,random_state=42,n_jobs=-1).importances_mean
        importance[key]=sorted([{"feature":f,"importance":float(v)} for f,v in zip(FEATURES,vals)],key=lambda z:z["importance"],reverse=True)
        all_metrics[key]=metrics; reports[key]=report; matrices[key]=cm; rocs[key]=roc; prs[key]=pr; predictions[key]=pred
        registry[key]={"display_name":NAMES[key],"artifact_filename":artifact.name,"macro_f1":metrics["macro_f1"],"training_timestamp":timestamp,
          "scikit_learn_version":sklearn.__version__,"python_version":platform.python_version(),"feature_order":FEATURES,"target_class_mapping":{str(k):v for k,v in CLASS_MAPPING.items()},"artifact_size_bytes":size}
        pd.DataFrame(report).transpose().to_csv(REPORTS/f"classification_report_{key}.csv")
    ranked=sorted(KEYS,key=lambda k:(-all_metrics[k]["macro_f1"],-all_metrics[k]["macro_recall"],all_metrics[k]["artifact_size_bytes"],all_metrics[k]["average_prediction_time_ms"]))
    best=ranked[0]
    for k in registry: registry[k]["is_best_model"]=(k==best)
    prep=json.loads((ART/"data_preparation_summary.json").read_text(encoding="utf-8"))
    metadata={"random_seed":42,"test_size":.2,"stratified":True,"feature_order":FEATURES,"target":TARGET,"classes":[0,1,2],
      "train_records":len(train_i),"test_records":len(test_i),"train_class_counts":native(ytrain.value_counts().sort_index().to_dict()),"test_class_counts":native(ytest.value_counts().sort_index().to_dict())}
    feature_meta={"feature_order":FEATURES,"target":TARGET,"class_mapping":{str(k):v for k,v in CLASS_MAPPING.items()}}
    best_doc={"model_key":best,"display_name":NAMES[best],"selection_metric":"macro_f1","macro_f1":all_metrics[best]["macro_f1"]}
    # Offline subgroup evaluation for the selected model, using only test-set attributes.
    subgroup={"note":"Subgroup results depend on dataset composition and are not evidence of clinical fairness.","groups":{}}
    testdf=df.iloc[test_i].copy(); testdf["prediction"]=predictions[best]
    if "Sex" in testdf:
      subgroup["groups"]["Sex"]={str(g):{"records":len(d),"macro_f1":f1_score(d[TARGET],d.prediction,average="macro",zero_division=0),"balanced_accuracy":balanced_accuracy_score(d[TARGET],d.prediction)} for g,d in testdf.groupby("Sex")}
    testdf["broad_age_group"]=pd.cut(testdf.Age,[0,5,9,13],labels=["18-44","45-64","65+"])
    subgroup["groups"]["Broad age"]={str(g):{"records":len(d),"macro_f1":f1_score(d[TARGET],d.prediction,average="macro",zero_division=0),"balanced_accuracy":balanced_accuracy_score(d[TARGET],d.prediction)} for g,d in testdf.groupby("broad_age_group",observed=True)}
    for path,obj in [(ART/"model_metrics.json",all_metrics),(ART/"classification_reports.json",reports),(ART/"confusion_matrices.json",matrices),(ART/"roc_curve_data.json",rocs),(ART/"precision_recall_data.json",prs),(ART/"feature_importance.json",importance),(ART/"training_metadata.json",metadata),(ART/"feature_metadata.json",feature_meta),(ART/"best_model.json",best_doc),(ART/"dataset_insights.json",insights(df,prep)),(ART/"subgroup_evaluation.json",subgroup),(MODELS/"model_registry.json",registry)]: dump(path,native(obj))
    pd.DataFrame([{"model":NAMES[k],**all_metrics[k]} for k in KEYS]).to_csv(REPORTS/"model_comparison.csv",index=False)
    y.value_counts().sort_index().rename_axis("class").reset_index(name="count").to_csv(REPORTS/"class_distribution.csv",index=False)
    print("\nTraining complete"); print(f"Best model: {NAMES[best]}")
    for k in KEYS: print(f"{NAMES[k]}: macro F1={all_metrics[k]['macro_f1']:.4f}, size={all_metrics[k]['artifact_size_bytes']:,} bytes")
if __name__=="__main__": main()
