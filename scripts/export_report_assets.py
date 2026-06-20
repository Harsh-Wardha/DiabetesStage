"""Generate report PNGs from authentic stored artifacts and the cleaned dataset."""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/"artifacts"; OUT=ROOT/"reports"; OUT.mkdir(exist_ok=True)
def load(n): return json.loads((ART/n).read_text(encoding="utf-8"))
def save(name): plt.tight_layout(); plt.savefig(OUT/name,dpi=150,bbox_inches="tight"); plt.close()
def main():
    metrics=load("model_metrics.json"); matrices=load("confusion_matrices.json"); imp=load("feature_importance.json"); best=load("best_model.json")["model_key"]
    keys=list(metrics); labels=[k.replace('_',' ').title() for k in keys]
    for filename,fields,title in [("model_comparison.png",["accuracy","balanced_accuracy","macro_f1"],"Model comparison"),("training_time_comparison.png",["training_time_seconds"],"Training time"),("model_size_comparison.png",["artifact_size_bytes"],"Artifact size")]:
        x=np.arange(len(keys)); width=.75/max(len(fields),1)
        for j,f in enumerate(fields): plt.bar(x+(j-(len(fields)-1)/2)*width,[metrics[k][f] for k in keys],width,label=f.replace('_',' ').title())
        plt.xticks(x,labels,rotation=15); plt.title(title); plt.legend(); save(filename)
    counts=pd.read_csv(ROOT/"reports/class_distribution.csv"); plt.bar(counts["class"].astype(str),counts["count"],color="#087f78"); plt.title("Class distribution"); save("class_distribution.png")
    for key in keys:
        for norm,suffix in [(False,""),(True,"_normalized")]:
            a=np.array(matrices[key]["normalized" if norm else "raw"]); plt.imshow(a,cmap="Blues"); plt.colorbar(); plt.title(f"{labels[keys.index(key)]} confusion matrix")
            for i in range(3):
                for j in range(3): plt.text(j,i,f"{a[i,j]:.2f}" if norm else str(int(a[i,j])),ha="center",va="center")
            plt.xlabel("Predicted"); plt.ylabel("Actual"); save(f"confusion_matrix_{key}{suffix}.png")
    top=imp[best][:12][::-1]; plt.barh([x["feature"] for x in top],[x["importance"] for x in top],color="#087f78"); plt.title("Global feature importance"); save("feature_importance.png")
    for source,filename,xk,yk,title in [("roc_curve_data.json","roc_curves.png","fpr","tpr","ROC curves"),("precision_recall_data.json","precision_recall_curves.png","recall","precision","Precision-recall curves")]:
        data=load(source)
        for key in keys:
            for c in ['0','1','2']: plt.plot(data[key][c][xk],data[key][c][yk],label=f"{labels[keys.index(key)]} class {c}",alpha=.75)
        plt.xlabel(xk.title()); plt.ylabel(yk.title()); plt.title(title); plt.legend(fontsize=7); save(filename)
    df=pd.read_csv(ROOT/"dataset/processed/diabetes_stage_clean.csv")
    plt.hist(df.BMI,bins=40,color="#087f78"); plt.title("BMI distribution"); save("bmi_distribution.png")
    print(f"Report assets written to {OUT}")
if __name__=="__main__": main()
