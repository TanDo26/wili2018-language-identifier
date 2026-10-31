from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}

def test_predict_success():
    res = client.post("/api/v1/predict", json={
        "text": "This is a sample sentence to test language identification system."
    })
    assert res.status_code == 200
    data = res.json()
    assert "language_code" in data
    assert "confidence" in data
    assert "is_unsure" in data

def test_predict_text_too_short():
    res = client.post("/api/v1/predict", json={"text": "Short text"})
    # Pydantic validation sẽ tự động chặn từ Schema min_length=20 -> trả mã 422 hoặc 400
    assert res.status_code in [400, 422]

def test_report_creation():
    res = client.post("/api/v1/report", json={
        "text": "Sample wrong prediction text",
        "predicted_language": "vie",
        "correct_language": "eng",
        "user_feedback": "Incorrect classification"
    })
    assert res.status_code == 201
    assert "report_id" in res.json()
