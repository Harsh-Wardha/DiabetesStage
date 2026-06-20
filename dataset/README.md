# Dataset

Use the real three-class CDC Diabetes Health Indicators BRFSS 2015 CSV. It must be named `diabetes_012_health_indicators_BRFSS2015.csv` and contain target classes 0, 1, and 2. Raw and processed records are excluded from deployment and Git. No synthetic fallback is permitted.

Import and validate a downloaded copy with `python scripts/fetch_dataset.py --input path/to/diabetes_012_health_indicators_BRFSS2015.csv`, then run `python scripts/prepare_dataset.py`.
