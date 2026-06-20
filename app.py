"""Flask entry point. Importing this module performs no training or writes."""
from __future__ import annotations
import logging, os
from flask import Flask, jsonify, render_template, request
from config import CLASS_MAPPING, Config, DISCLAIMER
from services.analytics_service import dashboard_data
from services.model_service import ModelUnavailableError, load_json
from services.prediction_service import predict as make_prediction
from services.validation_service import ValidationError

app=Flask(__name__,static_folder="public",static_url_path="")
app.config.from_object(Config)
logging.basicConfig(level=logging.INFO,format="%(levelname)s %(name)s %(message)s")
if os.getenv("VERCEL")=="1" and not os.getenv("SECRET_KEY"): app.logger.warning("SECRET_KEY is missing in production")

@app.context_processor
def globals_context(): return {"disclaimer":DISCLAIMER}
@app.get("/")
def index(): return render_template("index.html")
@app.route("/predict",methods=["GET","POST"])
def prediction():
    if request.method=="GET": return render_template("predict.html")
    try: return render_template("result.html",result=make_prediction(request.form.to_dict()))
    except ValidationError as exc: return render_template("predict.html",error=str(exc)),400
    except ModelUnavailableError as exc: return render_template("model_unavailable.html",message=str(exc)),503
@app.post("/api/predict")
def prediction_api():
    try: return jsonify(make_prediction(request.get_json(silent=True) or {}))
    except ValidationError as exc: return jsonify({"error":str(exc)}),400
    except ModelUnavailableError as exc: return jsonify({"error":str(exc)}),503
@app.get("/dashboard")
def dashboard():
    try: return render_template("dashboard.html",data=dashboard_data())
    except ModelUnavailableError as exc: return render_template("model_unavailable.html",message=str(exc)),503
@app.get("/insights")
def insights():
    try: return render_template("insights.html",data=load_json("dataset_insights.json"))
    except ModelUnavailableError as exc: return render_template("model_unavailable.html",message=str(exc)),503
@app.get("/methodology")
def methodology(): return render_template("methodology.html")
@app.get("/about")
def about(): return render_template("about.html")
@app.get("/health")
def health():
    try: best=load_json("best_model.json"); return jsonify({"status":"ok","models_available":True,"best_model":best["display_name"]})
    except ModelUnavailableError: return jsonify({"status":"degraded","models_available":False}),503
@app.errorhandler(404)
def not_found(error): return render_template("404.html"),404
@app.errorhandler(500)
def server_error(error): return render_template("500.html"),500
if __name__=="__main__": app.run(debug=True)
