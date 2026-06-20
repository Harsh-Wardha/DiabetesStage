"""Estimate repository files included in the Vercel bundle."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXCLUDED={".git",".venv","venv","tests","reports","__pycache__",".pytest_cache","dataset/raw","dataset/processed"}
def excluded(p):
    rel=p.relative_to(ROOT).as_posix()
    return (any(rel==x or rel.startswith(x+"/") for x in EXCLUDED)
            or p.name in {"diabetes_012_health_indicators_BRFSS2015.csv","training.stdout.log","training.stderr.log"}
            or p.suffix in {".pyc",".ipynb"}
            or p.name.startswith(".env") and p.name!=".env.example")
def main():
    files=[p for p in ROOT.rglob("*") if p.is_file() and not excluded(p)]; total=sum(p.stat().st_size for p in files)
    print(f"Estimated included repository files: {len(files)} files, {total/1024/1024:.2f} MiB")
    for folder in ["models","artifacts","templates","public","services"]:
        size=sum(p.stat().st_size for p in (ROOT/folder).rglob("*") if p.is_file()); print(f"{folder}: {size/1024/1024:.2f} MiB")
    if total>100*1024*1024: print("WARNING: bundle source exceeds 100 MiB; inspect large files.")
if __name__=="__main__": main()
