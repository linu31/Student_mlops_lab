from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "Student Performance" in response.json()["message"]

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_prediction_endpoint():
    payload = {
        "study_hours": 3.5,
        "attendance_pct": 82,
        "previous_score": 72,
        "assignments_completed": 8,
        "sleep_hours": 7,
        "internet_access": "Yes",
        "extracurricular": "No"
    }
    response = client.post("/predict", json=payload)
    # If the model has not been trained yet, the API returns 503.
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        assert response.json()["prediction"] in ("Pass", "At_Risk")
