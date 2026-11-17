from pathlib import Path
import json
import warnings
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from feature_pipeline import engineer_features

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/students_v2.csv"
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)
MODEL_NAME = "StudentPerformanceRiskModel"

def train():
    feature_store = ROOT / "data/processed/feature_store.csv"
    if feature_store.exists():
        df = pd.read_csv(feature_store)  # Reuse persisted engineered features
    else:
        df = engineer_features(pd.read_csv(DATA))
    y = df["performance"].map({"At_Risk": 1, "Pass": 0})
    X = df.drop(columns=["performance", "student_id"])
    numeric = ["study_hours", "attendance_pct", "previous_score",
               "assignments_completed", "sleep_hours", "assignment_rate",
               "study_effort", "academic_base"]
    categorical = ["internet_access", "extracurricular"]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical)
    ])
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=180, max_depth=10, random_state=42,
                                                class_weight="balanced")
    }
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment("Student_Performance_MLOps")
    results = {}
    best_name, best_score, best_pipeline = None, -1, None
    for name, model in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        with mlflow.start_run(run_name=name) as run:
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            prob = pipe.predict_proba(X_test)[:, 1]
            metrics = {
                "accuracy": float(accuracy_score(y_test, pred)),
                "precision": float(precision_score(y_test, pred, zero_division=0)),
                "recall": float(recall_score(y_test, pred, zero_division=0)),
                "f1": float(f1_score(y_test, pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test, prob))
            }
            mlflow.log_param("model_type", name)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipe, artifact_path="model")
            results[name] = metrics
            if metrics["f1"] > best_score:
                best_name, best_score, best_pipeline = name, metrics["f1"], pipe
    joblib.dump(best_pipeline, ART / "best_model.joblib")
    best_pred = best_pipeline.predict(X_test)
    report = {
        "best_model": best_name,
        "best_f1": best_score,
        "metrics": results,
        "classification_report": classification_report(y_test, best_pred, output_dict=True)
    }
    (ART / "training_results.json").write_text(json.dumps(report, indent=2))
    # Register the selected model in MLflow Model Registry.
    with mlflow.start_run(run_name=f"Register_{best_name}") as run:
        mlflow.log_param("selected_model", best_name)
        mlflow.log_metric("selected_f1", best_score)
        mlflow.sklearn.log_model(
            best_pipeline, artifact_path="registered_model",
            registered_model_name=MODEL_NAME
        )
    print(json.dumps(report, indent=2))
    print(f"Registered model name: {MODEL_NAME}")
    print("Best model saved to artifacts/best_model.joblib")
    return report

if __name__ == "__main__":
    train()
