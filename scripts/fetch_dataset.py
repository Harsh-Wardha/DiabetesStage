"""Import a manual CSV or try UCI dataset 891 without binary substitution."""
from __future__ import annotations
import argparse, shutil, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "dataset/raw/diabetes_012_health_indicators_BRFSS2015.csv"

def classes(path: Path) -> list[int]:
    df = pd.read_csv(path)
    targets = [c for c in df.columns if c.lower().replace("_", "") in {"diabetes012", "diabetesstatus"}]
    if not targets: raise ValueError("No three-class Diabetes_012 target column found.")
    values = sorted(pd.to_numeric(df[targets[0]], errors="coerce").dropna().astype(int).unique().tolist())
    if values != [0,1,2]: raise ValueError(f"Wrong dataset: expected target classes [0,1,2], found {values}.")
    return values

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--input", type=Path); args=parser.parse_args()
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if args.input:
        classes(args.input); shutil.copy2(args.input, DEST); print(f"Validated and copied to {DEST}"); return
    try:
        from ucimlrepo import fetch_ucirepo
        data=fetch_ucirepo(id=891); frame=pd.concat([data.data.features, data.data.targets], axis=1)
        target=data.data.targets.iloc[:,0]
        values=sorted(pd.to_numeric(target, errors="coerce").dropna().astype(int).unique().tolist())
        if values != [0,1,2]: raise ValueError(f"UCI response has target classes {values}, not [0,1,2]")
        frame.to_csv(DEST,index=False); classes(DEST); print(f"Saved validated UCI dataset to {DEST}")
    except Exception as exc:
        print(f"Could not retrieve a valid three-class file: {exc}\nPlace diabetes_012_health_indicators_BRFSS2015.csv in dataset/raw/ or run:\npython scripts/fetch_dataset.py --input path/to/diabetes_012_health_indicators_BRFSS2015.csv", file=sys.stderr)
        raise SystemExit(1)
if __name__ == "__main__": main()
