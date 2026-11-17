# Student Performance Risk Prediction – End-to-End MLOps

A complete academic MLOps demonstration: synthetic data versioning, preprocessing, feature engineering, model comparison, MLflow tracking/registry, GitHub Actions CI, and FastAPI serving.

## 1. Problem statement
Predict whether a student is likely to pass or is academically at risk using study hours, attendance, previous score, assignments, sleep hours, internet access and extracurricular participation.

## 2. Project structure
```text
Student_Performance_MLOps/
├── data/raw/                 # V1 and V2 synthetic CSV datasets
├── data/processed/           # Reusable feature store CSV
├── artifacts/                # Best model and evaluation results
├── src/
│   ├── generate_data.py      # Dataset creation/versioning
│   ├── feature_pipeline.py   # Feature engineering/store
│   ├── train.py              # preprocessing, training, MLflow
│   └── api.py                # FastAPI model serving
├── tests/test_api.py         # Automated API tests
├── .github/workflows/        # GitHub Actions CI
├── reports/                  # Implementation report
└── requirements.txt
```

## 3. Run locally
Use Python 3.10 or 3.11. In a terminal from this folder:
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/generate_data.py
python src/feature_pipeline.py
python src/train.py
pytest -q
uvicorn src.api:app --reload
```
Open `http://127.0.0.1:8000/health`. POST a JSON body to `/predict`. Interactive API docs are available at `/docs` (optional; not required by the exam).

## 4. MLflow
Training creates local tracking data under `mlruns/`, logs parameters/metrics/models for Logistic Regression and Random Forest, and registers the selected model as `StudentPerformanceRiskModel`. Start UI:
```bash
mlflow ui --backend-store-uri ./mlruns
```
Then open `http://127.0.0.1:5000`.

## 5. Example API request
```json
{
  "study_hours": 3.5,
  "attendance_pct": 82,
  "previous_score": 72,
  "assignments_completed": 8,
  "sleep_hours": 7,
  "internet_access": "Yes",
  "extracurricular": "No"
}
```

## 6. Dataset version notes
- V1: 1,200 synthetic student records.
- V2: 1,400 records (V1-sized base plus 200 additional synthetic records); same schema and generation logic.
- Missing values are deliberately introduced in numeric columns to demonstrate imputation.

## 7. Feature store
`data/processed/feature_store.csv` stores engineered columns `assignment_rate`, `study_effort`, and `academic_base`. Training reuses the same feature engineering function to avoid inconsistent feature definitions.

## 8. CI/CD
GitHub Actions installs dependencies, generates datasets, creates features, trains both models, registers the best model, and runs tests on pushes and pull requests. The workflow file is `.github/workflows/mlops-ci.yml`.

## 9. Important academic note
The dataset is synthetic and the metrics are for demonstration only. They are not evidence of real-world student outcomes. Run the commands in order and capture your own screenshots from your system and GitHub Actions.
