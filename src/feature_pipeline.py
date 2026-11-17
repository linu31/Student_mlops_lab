from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create reusable domain features without using the target column."""
    out = df.copy()
    out["assignment_rate"] = out["assignments_completed"] / 10.0
    out["study_effort"] = out["study_hours"] * out["attendance_pct"] / 100.0
    out["academic_base"] = (out["previous_score"] + out["attendance_pct"]) / 2.0
    return out

if __name__ == "__main__":
    source = ROOT / "data/raw/students_v2.csv"
    dest = ROOT / "data/processed/feature_store.csv"
    df = pd.read_csv(source)
    features = engineer_features(df)
    features.to_csv(dest, index=False)
    print(f"Feature store saved: {dest} ({len(features)} rows)")
