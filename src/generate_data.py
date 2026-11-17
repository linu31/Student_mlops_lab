from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

def make_dataset(n=1200, seed=42, version=1):
    rng = np.random.default_rng(seed)
    study = np.clip(rng.normal(3.2, 1.5, n), 0, 8)
    attendance = np.clip(rng.normal(78, 15, n), 25, 100)
    previous = np.clip(rng.normal(65, 16, n), 10, 100)
    assignments = rng.integers(0, 11, n)
    sleep = np.clip(rng.normal(7, 1.2, n), 3, 10)
    internet = rng.choice(["Yes", "No"], n, p=[0.82, 0.18])
    extra = rng.choice(["Yes", "No"], n, p=[0.45, 0.55])
    # Synthetic risk score; higher values indicate stronger academic performance.
    score = (study * 5.2 + attendance * 0.23 + previous * 0.30 +
             assignments * 1.45 + sleep * 1.2 +
             (internet == "Yes") * 2.0 + (extra == "Yes") * 1.0 +
             rng.normal(0, 9, n))
    target = np.where(score >= 52, "Pass", "At_Risk")
    df = pd.DataFrame({
        "student_id": [f"STU{i+1:04d}" for i in range(n)],
        "study_hours": study.round(1),
        "attendance_pct": attendance.round(1),
        "previous_score": previous.round(1),
        "assignments_completed": assignments,
        "sleep_hours": sleep.round(1),
        "internet_access": internet,
        "extracurricular": extra,
        "performance": target
    })
    # Add realistic missing values to demonstrate preprocessing.
    for col in ["study_hours", "attendance_pct", "previous_score", "sleep_hours"]:
        idx = rng.choice(df.index, size=max(1, int(n * 0.025)), replace=False)
        df.loc[idx, col] = np.nan
    if version == 2:
        # V2 adds a new batch of records and a small data-quality correction.
        extra_df = make_dataset(n=200, seed=202, version=1)
        extra_df["student_id"] = [f"STU{i+1201:04d}" for i in range(len(extra_df))]
        df = pd.concat([df, extra_df], ignore_index=True)
    return df

if __name__ == "__main__":
    v1 = make_dataset()
    v2 = make_dataset(version=2)
    v1.to_csv(RAW / "students_v1.csv", index=False)
    v2.to_csv(RAW / "students_v2.csv", index=False)
    print(f"Created V1: {len(v1)} rows; V2: {len(v2)} rows")
    print("V1 columns:", ", ".join(v1.columns))
