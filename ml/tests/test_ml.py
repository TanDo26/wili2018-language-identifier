"""Kiểm tra ml.train / ml.evaluate khớp với scripts/retrain.py và API, trên một bộ dữ liệu nhỏ cùng định dạng
data/processed/ (id, text, label; tập cắt ngắn có cột length; overlap_with_train.csv)."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline

import scripts.retrain as retrain
from api.main import app
from data.common import write_split
from ml import evaluate, naive_bayes, train
from ml.predict import predict

SENTENCES = {
    "eng": ["The river flows quietly through the old town every single morning.",
            "She reads a book every evening before she goes to bed.",
            "We are going to the market to buy some fresh fruit and bread.",
            "The weather today is warm and the sky is completely clear.",
            "He explained the difficult problem to the whole class again.",
            "My brother works at the hospital near the train station.",
            "They walked along the beach and watched the sunset together."],
    "vie": ["Con sông chảy lặng lẽ qua phố cổ vào mỗi buổi sáng sớm.",
            "Cô ấy đọc một cuốn sách mỗi tối trước khi đi ngủ.",
            "Chúng tôi đi chợ để mua một ít trái cây tươi và bánh mì.",
            "Hôm nay trời ấm và bầu trời rất trong xanh không một gợn mây.",
            "Anh ấy giải thích lại vấn đề khó cho cả lớp cùng nghe.",
            "Anh trai tôi làm việc ở bệnh viện gần nhà ga xe lửa.",
            "Họ đi dạo dọc bờ biển và cùng nhau ngắm hoàng hôn."],
    "fra": ["La rivière traverse doucement la vieille ville chaque matin.",
            "Elle lit un livre chaque soir avant de dormir dans sa chambre.",
            "Nous allons au marché pour acheter des fruits frais et du pain.",
            "Il fait chaud aujourd'hui et le ciel est complètement dégagé.",
            "Il a expliqué de nouveau le problème difficile à toute la classe.",
            "Mon frère travaille à l'hôpital près de la gare du centre.",
            "Ils ont marché le long de la plage en regardant le coucher du soleil."],
    "deu": ["Der Fluss fließt jeden Morgen ruhig durch die schöne Altstadt.",
            "Sie liest jeden Abend ein Buch, bevor sie schlafen geht.",
            "Wir gehen auf den Markt, um frisches Obst und Brot zu kaufen.",
            "Das Wetter ist heute warm und der Himmel ist völlig klar.",
            "Er erklärte der ganzen Klasse das schwierige Problem noch einmal.",
            "Mein Bruder arbeitet im Krankenhaus in der Nähe vom Bahnhof.",
            "Sie gingen am Strand entlang und sahen gemeinsam den Sonnenuntergang."],
}
SHORT = (20, 30, 50, 100)


def frame(split: str, rows) -> pd.DataFrame:
    return pd.DataFrame([{"id": f"{split}_{i:03d}", "text": t, "label": l} for i, (t, l) in enumerate(rows)])


def shorten(df: pd.DataFrame, sep: str) -> pd.DataFrame:
    return pd.DataFrame([{"id": f"{r.id}{sep}{n}", "text": r.text[:n].strip(), "label": r.label,
                          "source_id": r.id, "length": n} for r in df.itertuples() for n in SHORT])


@pytest.fixture
def data_dir(tmp_path):
    """data/processed/ thu nhỏ: 5 câu train, 1 val, 1 test mỗi ngôn ngữ."""
    path = tmp_path / "processed"
    rows = lambda part: [(t, lang) for lang, ss in SENTENCES.items() for t in ss[part]]
    splits = {"train": frame("tr", rows(slice(0, 5))), "val": frame("va", rows(slice(5, 6))),
              "test": frame("te", rows(slice(6, 7)))}
    for name, df in splits.items():
        write_split(df, path / f"{name}.csv")
    write_split(shorten(splits["train"], "#c").drop(columns="length"), path / "train_crops.csv")
    write_split(shorten(splits["val"], "@"), path / "val_short.csv")
    write_split(shorten(splits["test"], "@"), path / "test_short.csv")
    write_split(pd.DataFrame([{"id": "te_000", "split": "test", "shared_fraction": "0.5"}]),
                path / "overlap_with_train.csv")
    write_split(pd.DataFrame([{"id": "fb_1", "text": "Bonjour à tous, voici un nouveau texte en français.",
                               "label": "fra", "report_id": "1"}]), path / "train_feedback.csv")
    return path


def run_train(data_dir, out_dir, *extra):
    return train.main(["--data-dir", str(data_dir), "--out-dir", str(out_dir), *extra])


def test_algorithms_match_retrain():
    assert train.ALGORITHMS == retrain.ALGORITHMS


@pytest.mark.parametrize("algorithm", train.ALGORITHMS)
def test_models_plug_into_feature_search(algorithm):
    from experiments.feature_search import make_classifier
    clf = make_classifier(f"ml.{algorithm}:build_model")
    assert hasattr(clf, "fit") and hasattr(clf, "predict")


@pytest.mark.parametrize("algorithm", train.ALGORITHMS)
def test_train_and_evaluate_follow_retrain_contract(data_dir, tmp_path, algorithm):
    out = run_train(data_dir, tmp_path / "out", "--algorithm", algorithm, "--augment")

    model = joblib.load(out / "model.joblib")
    assert isinstance(model, Pipeline)
    proba = model.predict_proba(["the weather is warm today in the old town"])
    assert proba.shape == (1, 4) and np.isclose(proba.sum(), 1.0)
    assert sorted(model.classes_) == sorted(SENTENCES)

    metrics = evaluate.main(["--model-dir", str(out), "--data-dir", str(data_dir)])
    assert set(retrain.REQUIRED_FILES) <= {p.name for p in out.iterdir()}
    saved = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert saved == json.loads(json.dumps(metrics))
    assert all(0 <= saved[k] <= 1 for k in retrain.REQUIRED_METRICS)
    assert list(saved["by_length"]) == ["20", "30", "50", "100", "full"]
    assert saved["by_length"]["full"]["n"] == saved["n"] == 4
    assert saved["no_overlap"]["n"] == 3  # te_000 có văn bản chung với train
    assert {"top_confused_pairs", "mixed_language_check", "latency_ms_median", "latency_ms_p95"} <= set(saved)

    info = json.loads((out / "train_info.json").read_text(encoding="utf-8"))
    assert info["algorithm"] == algorithm
    assert info["n_train"] == 20 + 20 * len(SHORT)  # train + các đoạn cắt ngắn


def test_feedback_rows_are_added(data_dir, tmp_path):
    out = run_train(data_dir, tmp_path / "out", "--feedback")
    assert json.loads((out / "train_info.json").read_text(encoding="utf-8"))["n_train"] == 21


def test_tune_picks_value_from_grid(data_dir, tmp_path):
    out = run_train(data_dir, tmp_path / "out", "--algorithm", "naive_bayes", "--tune")
    info = json.loads((out / "train_info.json").read_text(encoding="utf-8"))
    assert [t["value"] for t in info["tuning"]] == list(naive_bayes.GRID)
    assert info["value"] == max(info["tuning"], key=lambda t: t["score"])["value"]


def test_nb_model_is_slimmed(data_dir, tmp_path):
    out = run_train(data_dir, tmp_path / "out", "--algorithm", "naive_bayes")
    clf = joblib.load(out / "model.joblib").named_steps["classifier"]
    assert not hasattr(clf, "feature_count_")
    assert clf.feature_log_prob_.dtype == np.float32
    assert clf.feature_log_prob_.flags["F_CONTIGUOUS"]  # joblib giữ thứ tự khi lưu / nạp lại


def test_svm_confidence_is_a_probability(data_dir, tmp_path):
    out = run_train(data_dir, tmp_path / "out", "--algorithm", "svm")
    result = predict(SENTENCES["vie"][0], out)[0]
    assert result["label"] == "vie"
    # softmax của decision_function chỉ cho khoảng 1/số lớp; sau hiệu chỉnh phải rõ ràng cao hơn
    assert result["confidence"] > 0.5


def test_missing_augmented_files_give_clear_error(data_dir, tmp_path):
    (data_dir / "train_crops.csv").unlink()
    with pytest.raises(SystemExit, match="data.pipeline"):
        run_train(data_dir, tmp_path / "out", "--augment")


def test_retrain_to_api_end_to_end(data_dir, tmp_path, monkeypatch):
    """scripts/retrain.py -> python -m ml.train -> python -m ml.evaluate -> models/v1 -> API."""
    monkeypatch.setattr(retrain, "PROCESSED_DIR", data_dir)
    models_dir = tmp_path / "models"
    version = retrain.main(["--skip-data", "--models-dir", str(models_dir), "--algorithm", "naive_bayes"])
    assert version.name == "v1"
    meta = json.loads((version / "meta.json").read_text(encoding="utf-8"))
    assert meta["algorithm"] == "naive_bayes" and set(meta["metrics"]) == {"accuracy", "macro_f1"}

    monkeypatch.setenv("MODELS_DIR", str(models_dir))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "app.db"))
    monkeypatch.setenv("ADMIN_TOKEN", "test-token")
    with TestClient(app) as client:
        res = client.post("/api/v1/predict", json={"text": "Wir gehen heute in die Altstadt und kaufen Brot."})
        assert res.status_code == 200
        assert (res.json()["language_code"], res.json()["language_name"]) == ("deu", "German")
        assert res.json()["model_version"] == "v1"
        listed = client.get("/api/v1/models").json()
        assert listed[0]["accuracy"] == meta["metrics"]["accuracy"]
        assert listed[0]["algorithm"] == "naive_bayes"
