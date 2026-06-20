"""Validate and clean the real three-class CDC indicator CSV."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from config import BINARY_FEATURES, FEATURES, TARGET

RAW = ROOT / "dataset/raw/diabetes_012_health_indicators_BRFSS2015.csv"
OUT = ROOT / "dataset/processed/diabetes_stage_clean.csv"
SUMMARY = ROOT / "artifacts/data_preparation_summary.json"
ALIASES = {"diabetes_012": TARGET, "diabetes012": TARGET, "diabetes_status": TARGET}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    lookup = {c.lower().replace(" ", "").replace("-", "").replace("_", ""): c for c in df.columns}
    expected = FEATURES + [TARGET, "Sex", "Education", "Income", "AnyHealthcare", "NoDocbcCost"]
    renames = {}
    for name in expected:
        key = name.lower().replace("_", "")
        if key in lookup: renames[lookup[key]] = name
    for old, new in ALIASES.items():
        if old in df.columns: renames[old] = new
    return df.rename(columns=renames)

def validate_three_classes(series: pd.Series) -> list[int]:
    classes = sorted(pd.to_numeric(series, errors="coerce").dropna().astype(int).unique().tolist())
    if classes == [0, 1]:
        raise ValueError("Binary target {0,1} detected. Use diabetes_012_health_indicators_BRFSS2015.csv with classes 0, 1 and 2.")
    if classes != [0, 1, 2]: raise ValueError(f"Target must contain exactly {{0,1,2}}; found {classes}.")
    return classes

def prepare(raw: Path = RAW, output: Path = OUT) -> dict:
    if not raw.exists(): raise FileNotFoundError(f"Real dataset not found: {raw}")
    df = normalize_columns(pd.read_csv(raw))
    missing = [c for c in FEATURES + [TARGET] if c not in df]
    if missing: raise ValueError(f"Missing required columns: {', '.join(missing)}")
    validate_three_classes(df[TARGET])
    original = len(df)
    duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()
    numeric_columns = list(dict.fromkeys(FEATURES + [TARGET, "Sex", "Education", "Income", "AnyHealthcare", "NoDocbcCost"]))
    numeric_columns = [c for c in numeric_columns if c in df]
    for c in numeric_columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    invalid = pd.Series(False, index=df.index)
    reasons = {}
    checks = {**{c: df[c].isin([0, 1]) for c in BINARY_FEATURES},
              "BMI": df.BMI.between(10, 100), "GenHlth": df.GenHlth.between(1, 5),
              "MentHlth": df.MentHlth.between(0, 30), "PhysHlth": df.PhysHlth.between(0, 30),
              "Age": df.Age.between(1, 13), TARGET: df[TARGET].isin([0, 1, 2])}
    for field, valid in checks.items():
        bad = ~valid.fillna(False); reasons[field] = int(bad.sum()); invalid |= bad
    invalid_count = int(invalid.sum())
    cleaned = df.loc[~invalid].copy()
    validate_three_classes(cleaned[TARGET])
    for c in BINARY_FEATURES + ["GenHlth", "MentHlth", "PhysHlth", "Age", TARGET]: cleaned[c] = cleaned[c].astype(int)
    output.parent.mkdir(parents=True, exist_ok=True); cleaned.to_csv(output, index=False)
    summary = {"source": raw.name, "original_rows": original, "duplicate_rows_removed": duplicates,
               "invalid_rows_removed": invalid_count, "invalid_counts_by_field": reasons,
               "cleaned_rows": len(cleaned), "columns": cleaned.columns.tolist(),
               "class_counts": {str(k): int(v) for k,v in cleaned[TARGET].value_counts().sort_index().items()}}
    SUMMARY.parent.mkdir(parents=True, exist_ok=True); SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

if __name__ == "__main__":
    try:
        result = prepare(); print(json.dumps(result, indent=2)); print(f"Clean dataset saved to {OUT}")
    except Exception as exc: print(f"Dataset preparation failed: {exc}", file=sys.stderr); raise SystemExit(1)

