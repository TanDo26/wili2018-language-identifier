import json
import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

SAMPLES = {
    "eng": "the quick brown fox jumps over the lazy dog every morning",
    "fra": "le renard brun saute par dessus le chien paresseux chaque matin",
    "vie": "con cao nau nhanh nhen nhay qua con cho luoi moi buoi sang",
}

@pytest.fixture
def client(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    version_dir = models_dir / "v1"
    version_dir.mkdir(parents=True)

    model = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3)), MultinomialNB())
    model.fit(list(SAMPLES.values()), list(SAMPLES.keys()))
    joblib.dump(model, version_dir / "model.joblib")
    
    (version_dir / "metrics.json").write_text(json.dumps({"accuracy": 0.9, "macro_f1": 0.9}))
    (version_dir / "meta.json").write_text(json.dumps({"version": "v1", "algorithm": "Multinomial Naive Bayes"}))

    monkeypatch.setenv("MODELS_DIR", str(models_dir))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "app.db"))
    monkeypatch.setenv("ADMIN_TOKEN", "test-token")

    from api.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "active_version" in data


def test_predict_success(client):
    res = client.post("/api/v1/predict", json={
        "text": "This is a sample sentence to test language identification system."
    })
    assert res.status_code == 200
    data = res.json()
    assert "prediction_id" in data
    assert "model_version" in data
    assert "language_code" in data
    assert "confidence" in data


def test_predict_validation_errors(client):
    # Short text -> 400
    res = client.post("/api/v1/predict", json={"text": "Short text"})
    assert res.status_code == 400
    assert "detail" in res.json()

    # Oversized text -> 413
    long_text = "a" * 5001
    res = client.post("/api/v1/predict", json={"text": long_text})
    assert res.status_code == 413
    assert "detail" in res.json()


def test_report_flow(client):
    # 1. Generate prediction
    pred_res = client.post("/api/v1/predict", json={
        "text": "This is a valid test sentence for predictions."
    })
    prediction_id = pred_res.json()["prediction_id"]

    # 2. Submit report
    rep_res = client.post("/api/v1/report", json={
        "prediction_id": prediction_id,
        "reason": "wrong_language",
        "suggested_language": "eng",
        "comment": "Incorrect classification"
    })
    assert rep_res.status_code == 201
    report_id = rep_res.json()["report_id"]

    # 3. Get reports (Admin)
    admin_res = client.get("/api/v1/reports", headers={"X-Admin-Token": "test-token"})
    assert admin_res.status_code == 200
    assert len(admin_res.json()) > 0

    # 4. Patch report (Admin)
    patch_res = client.patch(f"/api/v1/reports/{report_id}", 
                             json={"status": "accepted", "verified_language": "eng"},
                             headers={"X-Admin-Token": "test-token"})
    assert patch_res.status_code == 200
