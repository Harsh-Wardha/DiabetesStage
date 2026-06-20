# DiabetesStage: A Machine Learning-Based Diabetes Status Classification System

DiabetesStage is a university machine-learning project that compares three classifiers on the real three-class CDC Diabetes Health Indicators BRFSS 2015 dataset. Its Flask interface maps 16 indicators to the dataset-derived categories **No Diabetes Pattern (0)**, **Prediabetes Pattern (1)**, or **Diabetes Pattern (2)**.

> **Educational disclaimer:** This application is an educational machine-learning demonstration. It does not diagnose diabetes, prediabetes or any other medical condition. Only a qualified healthcare professional using appropriate clinical tests can make a diagnosis.

## Features and privacy

- Validates and cleans real data; binary datasets are rejected and no synthetic fallback exists.
- Trains Logistic Regression, Decision Tree, and Histogram Gradient Boosting on one shared stratified 80/20 split.
- Reports accuracy, balanced accuracy, macro/weighted/per-class metrics, ROC-AUC, confusion matrices, ROC and precision-recall curves.
- Selects the best model by macro F1, followed by macro recall, artifact size, and inference speed.
- Produces global feature importance and offline Sex/broad-age subgroup evaluation.
- Provides a responsive prediction form, evidence dashboard, dataset insights, and methodology pages.
- Uses pretrained, read-only artifacts on Vercel: no accounts, database, submitted-data storage, external ML API, runtime training, or dataset access.

## Dataset and methodology

The expected file is `diabetes_012_health_indicators_BRFSS2015.csv`, approximately 253,680 records, with target classes 0, 1, and 2. Inputs are HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, GenHlth, MentHlth, PhysHlth, DiffWalk, and Age.

Sex, Education, Income, AnyHealthcare, and NoDocbcCost are excluded from prediction to simplify the application and reduce dependence on sensitive demographic and socioeconomic attributes. Sex may be used only for aggregate offline evaluation. Cleaning normalizes columns, converts numeric values, removes exact duplicates, validates defined ranges, and documents every removal. Raw and processed CSVs are excluded from Git and Vercel.

`train_test_split(test_size=0.20, random_state=42, stratify=y)` creates one shared split. Logistic regression and decision tree use balanced class weights; histogram boosting receives balanced sample weights only during training. The untouched test set supplies reported metrics. Feature importance uses coefficient magnitude, tree importance, or reproducible held-out permutation importance; none establishes causation.

## Pages

- `/` overview; `/predict` stateless form; `/dashboard` actual model artifacts
- `/insights` prepared statistics; `/methodology` transparent process; `/health` deployment status

Prediction inputs exist only for the request and are not logged, written, or stored in browser local storage.

## Install, prepare, train, and run

Windows PowerShell:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
python scripts/fetch_dataset.py --input path\to\diabetes_012_health_indicators_BRFSS2015.csv
python scripts/prepare_dataset.py
python train_models.py
python scripts/export_report_assets.py
python scripts/verify_artifacts.py
pytest
python app.py
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
python scripts/fetch_dataset.py --input /path/to/diabetes_012_health_indicators_BRFSS2015.csv
python scripts/prepare_dataset.py
python train_models.py
python scripts/export_report_assets.py
python scripts/verify_artifacts.py
pytest
python app.py
```

Open <http://127.0.0.1:5000>. Training is never invoked by `app.py`.

## Architecture, deployment, and report

`scripts/` owns offline data and artifact tasks; `services/` owns validation, cached read-only model loading, inference, explanations, and analytics; `models/` and `artifacts/` are production inputs; `templates/` and `public/` are the UI. Reports and full data stay local. Vercel imports `app` from `app.py`; warm functions use `lru_cache` and prediction makes no network calls or writes. See [VERCEL_DEPLOYMENT.md](VERCEL_DEPLOYMENT.md).

The source is self-reported survey data from a particular US population and period. Outputs are not diagnoses, clinical probabilities, individualized advice, causation, or proof of fairness. Class imbalance, population shift, missing clinical measurements, and model error limit generalization. Future work could add external/temporal validation, calibration, uncertainty intervals, and community-reviewed fairness analysis; clinical use would require governance, prospective validation, and regulatory review.

For a university report, use authentic CSV/PNG assets in `reports/` to discuss cleaning, split design, imbalance, per-class errors, importance, subgroup limitations, privacy, and deployment. Screenshot placeholders: home page, prediction result, dashboard, and insights.

