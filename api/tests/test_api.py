import json
import uuid

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from api.export_openapi import OPENAPI_PATH, render
from api.main import app
from data.export_reports import reports_to_feedback

ADMIN = {"X-Admin-Token": "test-token"}
OK_TEXT = "the lazy dog sleeps while the quick brown fox jumps every morning"
SAMPLES = {
    "eng": "the quick brown fox jumps over the lazy dog every morning",
    "fra": "le renard brun saute par dessus le chien paresseux chaque matin",
    "vie": "con cáo nâu nhanh nhẹn nhảy qua con chó lười mỗi buổi sáng",
}


def add_version(models_dir, version, accuracy=0.9):
    """Tạo models/<version>/ giống scripts/retrain.py với một Pipeline nhỏ."""
    version_dir = models_dir / version
    version_dir.mkdir(parents=True)
    model = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3)), MultinomialNB(alpha=0.01))
    model.fit(list(SAMPLES.values()), list(SAMPLES))
    joblib.dump(model, version_dir / "model.joblib")
    (version_dir / "metrics.json").write_text(json.dumps({"accuracy": accuracy, "macro_f1": accuracy - 0.01}))
    (version_dir / "meta.json").write_text(json.dumps(
        {"version": version, "algorithm": "naive_bayes", "created_at": "2026-11-11T00:00:00+00:00"}))


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    path = tmp_path / "models"
    path.mkdir()
    monkeypatch.setenv("MODELS_DIR", str(path))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "app.db"))
    monkeypatch.setenv("ADMIN_TOKEN", "test-token")
    monkeypatch.delenv("UNSURE_THRESHOLD", raising=False)
    return path


@pytest.fixture
def client(models_dir):
    add_version(models_dir, "v1")
    with TestClient(app) as c:
        yield c


def predict_id(client) -> str:
    return client.post("/api/v1/predict", json={"text": OK_TEXT}).json()["prediction_id"]


def statuses(client) -> dict:
    return {m["version"]: m["status"] for m in client.get("/api/v1/models").json()}


# -----------------------------------------------------------------------------
# Health, predict, validation
# -----------------------------------------------------------------------------
def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "model_loaded": True, "active_version": "v1"}


def test_starts_without_models_and_predict_returns_503(models_dir):
    with TestClient(app) as c:
        assert c.get("/health").json() == {"status": "ok", "model_loaded": False, "active_version": None}
        res = c.post("/api/v1/predict", json={"text": OK_TEXT})
        assert res.status_code == 503
        assert isinstance(res.json()["detail"], str)


def test_predict_success(client):
    res = client.post("/api/v1/predict", json={"text": OK_TEXT})
    assert res.status_code == 200
    data = res.json()
    uuid.UUID(data["prediction_id"])
    assert data["model_version"] == "v1"
    assert (data["language_code"], data["language_name"]) == ("eng", "English")
    assert data["top_candidates"][0]["code"] == "eng"
    assert len(data["top_candidates"]) == 3
    assert data["is_unsure"] == (data["confidence"] < 0.8)


def test_unsure_threshold_from_env(client, monkeypatch):
    monkeypatch.setenv("UNSURE_THRESHOLD", "1.0")
    data = client.post("/api/v1/predict", json={"text": OK_TEXT}).json()
    assert data["is_unsure"] is True
    assert any("threshold" in line for line in data["explanation"])


def test_predict_batch(client):
    res = client.post("/api/v1/predict-batch", json={"texts": [OK_TEXT, SAMPLES["fra"]]})
    assert res.status_code == 200
    results = res.json()["results"]
    assert [r["language_code"] for r in results] == ["eng", "fra"]
    assert results[0]["prediction_id"] != results[1]["prediction_id"]


@pytest.mark.parametrize("path, body, expected", [
    ("/api/v1/predict", {"text": "Short text"}, 400),
    ("/api/v1/predict", {"text": " " * 30}, 400),
    ("/api/v1/predict", {"text": "a" * 5001}, 413),
    ("/api/v1/predict", {}, 400),
    ("/api/v1/predict", {"text": 123}, 400),
    ("/api/v1/predict-batch", {"texts": []}, 400),
    ("/api/v1/predict-batch", {"texts": [OK_TEXT] * 101}, 413),
    ("/api/v1/predict-batch", {"texts": [OK_TEXT, "a" * 5001]}, 413),
    ("/api/v1/predict-batch", {"texts": [OK_TEXT, "short"]}, 400),
])
def test_validation_errors(client, path, body, expected):
    res = client.post(path, json=body)
    assert res.status_code == expected
    assert isinstance(res.json()["detail"], str)


def test_text_limits_count_after_strip(client):
    assert client.post("/api/v1/predict", json={"text": "  " + "a" * 5000 + "  "}).status_code == 200
    assert client.post("/api/v1/predict", json={"text": "  " + "a" * 19 + "  "}).status_code == 400


def test_languages(client):
    languages = client.get("/api/v1/languages").json()
    assert len(languages) == 235
    assert {"code": "eng", "name": "English"} in languages


# -----------------------------------------------------------------------------
# Báo cáo
# -----------------------------------------------------------------------------
def test_report_flow(client):
    prediction_id = predict_id(client)
    res = client.post("/api/v1/report", json={
        "prediction_id": prediction_id,
        "reason": "wrong_language",
        "suggested_language": "fra",
        "comment": "Incorrect classification",
    })
    assert res.status_code == 201
    report_id = res.json()["report_id"]

    items = client.get("/api/v1/reports", headers=ADMIN).json()
    assert len(items) == 1
    item = items[0]
    assert item["report_id"] == report_id
    assert (item["text"], item["predicted_language"], item["model_version"]) == (OK_TEXT, "eng", "v1")
    assert (item["status"], item["verified_language"]) == ("new", None)

    res = client.patch(f"/api/v1/reports/{report_id}", json={"status": "accepted", "verified_language": "eng"},
                       headers=ADMIN)
    assert res.status_code == 200

    accepted = client.get("/api/v1/reports?status=accepted", headers=ADMIN).json()
    assert [r["report_id"] for r in accepted] == [report_id]
    assert client.get("/api/v1/reports?status=rejected", headers=ADMIN).json() == []
    assert client.get("/api/v1/reports?model_version=v9", headers=ADMIN).json() == []
    # data/export_reports.py đọc được kết quả để đưa vào retrain
    assert reports_to_feedback(accepted).to_dict("records") == [
        {"text": OK_TEXT, "label": "eng", "report_id": str(report_id)}]


@pytest.mark.parametrize("prediction_id, extra, expected", [
    ("unknown", {"reason": "other"}, 404),
    (None, {"reason": "other"}, 400),
    ("valid", {"reason": "bad_reason"}, 400),
    ("valid", {"reason": "other", "suggested_language": "xxx"}, 400),
    ("valid", {"reason": "other", "comment": "a" * 501}, 400),
])
def test_report_errors(client, prediction_id, extra, expected):
    body = dict(extra)
    if prediction_id == "valid":
        body["prediction_id"] = predict_id(client)
    elif prediction_id:
        body["prediction_id"] = prediction_id
    res = client.post("/api/v1/report", json=body)
    assert res.status_code == expected
    assert isinstance(res.json()["detail"], str)


def test_update_report_errors(client):
    report_id = client.post("/api/v1/report", json={"prediction_id": predict_id(client), "reason": "other"}).json()[
        "report_id"]
    patch = lambda rid, body: client.patch(f"/api/v1/reports/{rid}", json=body, headers=ADMIN)
    assert patch(report_id, {"status": "accepted"}).status_code == 400
    assert patch(report_id, {"status": "accepted", "verified_language": "xxx"}).status_code == 400
    assert patch(report_id, {"status": "done"}).status_code == 400
    assert patch(999, {"status": "rejected"}).status_code == 404
    assert patch(report_id, {"status": "rejected"}).status_code == 200


# -----------------------------------------------------------------------------
# Xác thực admin
# -----------------------------------------------------------------------------
ADMIN_CALLS = [
    ("get", "/api/v1/reports", None),
    ("patch", "/api/v1/reports/1", {"status": "rejected"}),
    ("post", "/api/v1/models/promote", {"version": "v1"}),
    ("post", "/api/v1/models/rollback", None),
]


@pytest.mark.parametrize("method, path, body", ADMIN_CALLS)
def test_admin_requires_token(client, method, path, body):
    for headers in ({}, {"X-Admin-Token": "wrong-token"}):
        res = client.request(method, path, json=body, headers=headers)
        assert res.status_code == 401
        assert isinstance(res.json()["detail"], str)


@pytest.mark.parametrize("method, path, body", ADMIN_CALLS)
def test_admin_disabled_without_admin_token(models_dir, monkeypatch, method, path, body):
    monkeypatch.delenv("ADMIN_TOKEN")
    add_version(models_dir, "v1")
    with TestClient(app) as c:
        res = c.request(method, path, json=body, headers={"X-Admin-Token": ""})
        assert res.status_code == 503


# -----------------------------------------------------------------------------
# Version mô hình
# -----------------------------------------------------------------------------
def test_list_models_reads_metrics_and_meta(client):
    assert client.get("/api/v1/models").json() == [{
        "version": "v1", "algorithm": "naive_bayes", "status": "active",
        "created_at": "2026-11-11T00:00:00+00:00", "accuracy": 0.9, "macro_f1": 0.89,
    }]


def test_first_start_uses_newest_version(models_dir):
    for version in ("v1", "v2", "v10"):
        add_version(models_dir, version)
    (models_dir / "draft").mkdir()  # không đúng dạng vN: bỏ qua
    with TestClient(app) as c:
        assert c.get("/health").json()["active_version"] == "v10"
        assert [m["version"] for m in c.get("/api/v1/models").json()] == ["v1", "v2", "v10"]


def test_promote_and_rollback(client, models_dir):
    add_version(models_dir, "v2", accuracy=0.95)  # retrain tạo version mới khi API đang chạy
    assert statuses(client) == {"v1": "active", "v2": "candidate"}

    promote = lambda version: client.post("/api/v1/models/promote", json={"version": version}, headers=ADMIN)
    assert promote("v9").status_code == 404
    assert promote("..").status_code == 404
    assert promote("v1").status_code == 409

    res = promote("v2")
    assert res.status_code == 200
    assert res.json()["active_version"] == "v2"
    assert client.post("/api/v1/predict", json={"text": OK_TEXT}).json()["model_version"] == "v2"
    assert statuses(client) == {"v1": "archived", "v2": "active"}

    res = client.post("/api/v1/models/rollback", headers=ADMIN)
    assert res.status_code == 200
    assert res.json()["active_version"] == "v1"
    assert client.post("/api/v1/predict", json={"text": OK_TEXT}).json()["model_version"] == "v1"
    assert statuses(client) == {"v1": "active", "v2": "archived"}
    assert client.post("/api/v1/models/rollback", headers=ADMIN).status_code == 409


def test_model_state_survives_restart(models_dir):
    add_version(models_dir, "v1")
    with TestClient(app) as c:
        add_version(models_dir, "v2")
        assert c.post("/api/v1/models/promote", json={"version": "v2"}, headers=ADMIN).status_code == 200

    with TestClient(app) as c:  # khởi động lại, cùng DB
        assert c.get("/health").json()["active_version"] == "v2"
        assert statuses(c) == {"v1": "archived", "v2": "active"}
        assert c.post("/api/v1/models/rollback", headers=ADMIN).json()["active_version"] == "v1"
        add_version(models_dir, "v3")
        assert c.post("/api/v1/models/promote", json={"version": "v3"}, headers=ADMIN).status_code == 200


# -----------------------------------------------------------------------------
# Hợp đồng API
# -----------------------------------------------------------------------------
def test_openapi_yaml_is_up_to_date():
    current = OPENAPI_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert current == render(app.openapi()), "openapi.yaml đã cũ: chạy python -m api.export_openapi"


def test_openapi_has_no_422_responses():
    for path_item in app.openapi()["paths"].values():
        for operation in path_item.values():
            assert "422" not in operation["responses"]
