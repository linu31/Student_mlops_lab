from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts/best_model.joblib"
app = FastAPI(title="Student Performance Risk Prediction API",
              description="Predicts whether a student is academically at risk.", version="1.0")

class StudentInput(BaseModel):
    study_hours: float = Field(ge=0, le=24)
    attendance_pct: float = Field(ge=0, le=100)
    previous_score: float = Field(ge=0, le=100)
    assignments_completed: int = Field(ge=0, le=10)
    sleep_hours: float = Field(ge=0, le=24)
    internet_access: str
    extracurricular: str

def get_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run python src/train.py first.")
    return joblib.load(MODEL_PATH)

@app.get("/")
def home():
    return {"message": "Student Performance Risk API is running", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_exists": MODEL_PATH.exists()}

@app.post("/predict")
def predict(student: StudentInput):
    try:
        model = get_model()
        row = pd.DataFrame([student.model_dump()])
        row["assignment_rate"] = row["assignments_completed"] / 10.0
        row["study_effort"] = row["study_hours"] * row["attendance_pct"] / 100.0
        row["academic_base"] = (row["previous_score"] + row["attendance_pct"]) / 2.0
        prediction = int(model.predict(row)[0])
        probability = float(model.predict_proba(row)[0][1])
        return {
            "prediction": "At_Risk" if prediction == 1 else "Pass",
            "risk_probability": round(probability, 4),
            "model": "best registered/trained pipeline"
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
