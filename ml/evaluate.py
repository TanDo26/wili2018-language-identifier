"""Đánh giá model.joblib trên tập test của nhóm, ghi metrics.json vào cùng thư mục.

    make retrain                                   # scripts/retrain.py gọi file này sau ml.train
    python -m ml.evaluate --model-dir experiments/output/ml/nb

metrics.json (scripts/retrain.py và GET /api/v1/models đọc accuracy, macro_f1):
  accuracy, macro_f1, n   trên test.csv
  by_length               {"20", "30", "50", "100", "full"} -> {accuracy, macro_f1, n}, từ test_short.csv và test.csv
  no_overlap              {accuracy, macro_f1, n} trên các đoạn test không có văn bản chung với train
                          (overlap_with_train.csv của data.check_leakage)
  top_confused_pairs      15 cặp (nhãn đúng -> nhãn đoán) nhầm nhiều nhất trên test.csv
  mixed_language_check    ghép 2 đoạn test khác ngôn ngữ: tỉ lệ dự đoán là một trong hai ngôn ngữ
  latency_ms_median/p95   thời gian dự đoán 1 đoạn như API (chuẩn hóa + vectorize + predict_proba)
"""
import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from data.common import (
    OVERLAP_PATH,
    PROCESSED_DIR,
    SEED,
    TEST_SHORT_PATH,
    load_split,
    normalize_text,
    read_split,
    rel,
    setup_console,
    write_json,
)
from ml.train import scores_by_bucket, with_buckets

CHUNK = 50_000  # dự đoán theo khối để ~570k đoạn test không chiếm hết RAM


def predict_in_chunks(model, texts: pd.Series) -> np.ndarray:
    return np.concatenate([model.predict(texts.iloc[i:i + CHUNK]) for i in range(0, len(texts), CHUNK)])


def scores(y_true, y_pred) -> dict:
    return {"accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
            "n": int(len(y_true))}


def confused_pairs(y_true, y_pred, top: int = 15) -> list[dict]:
    wrong = pd.DataFrame({"true": y_true, "pred": y_pred})[y_true != y_pred]
    counts = wrong.value_counts().head(top)
    return [{"true": t, "pred": p, "count": int(c)} for (t, p), c in counts.items()]


def mixed_language_check(model, test: pd.DataFrame, n_pairs: int = 500) -> dict:
    rng = np.random.default_rng(SEED)
    i, j = rng.integers(len(test), size=(2, n_pairs * 2))
    labels = test["label"].to_numpy()
    keep = labels[i] != labels[j]
    i, j = i[keep][:n_pairs], j[keep][:n_pairs]
    texts = test["text"].to_numpy()
    pred = model.predict([f"{a} {b}" for a, b in zip(texts[i], texts[j])])
    hits = (pred == labels[i]) | (pred == labels[j])
    return {"pairs": int(len(i)), "prediction_is_one_of_the_two": round(float(hits.mean()), 4)}


def latency(model, texts, n: int = 200) -> dict:
    times = []
    for text in texts[:n]:
        start = time.perf_counter()
        model.predict_proba([normalize_text(text)])
        times.append((time.perf_counter() - start) * 1000)
    return {"latency_ms_median": round(float(np.median(times)), 2),
            "latency_ms_p95": round(float(np.percentile(times, 95)), 2)}


def evaluate(model_dir: Path, data_dir: Path) -> dict:
    t0 = time.time()
    model = joblib.load(model_dir / "model.joblib")
    test = load_split("test", data_dir)
    short_path = data_dir / TEST_SHORT_PATH.name
    if not short_path.exists():
        raise SystemExit(f"Không thấy {rel(short_path)}: chạy `python -m data.pipeline` trước.")
    combined = with_buckets(test, read_split(short_path))
    pred = predict_in_chunks(model, combined["text"])

    is_full = (combined["bucket"] == "full").to_numpy()
    y_test, pred_test = test["label"].to_numpy(), pred[is_full]
    metrics = scores(y_test, pred_test)
    by_length = scores_by_bucket(combined, pred)
    by_length.pop("score")
    metrics["by_length"] = by_length

    overlap_path = data_dir / OVERLAP_PATH.name
    if overlap_path.exists():
        overlap = read_split(overlap_path)
        seen = set(overlap.loc[overlap["split"] == "test", "id"])
        clean = ~test["id"].isin(seen).to_numpy()
        metrics["no_overlap"] = scores(y_test[clean], pred_test[clean])
    else:
        print(f"Không thấy {rel(overlap_path)}: bỏ qua no_overlap (chạy `python -m data.check_leakage`).")

    metrics["top_confused_pairs"] = confused_pairs(y_test, pred_test)
    metrics["mixed_language_check"] = mixed_language_check(model, test)
    metrics.update(latency(model, test["text"].tolist()))
    metrics["seconds_total"] = round(time.time() - t0, 1)

    write_json(metrics, model_dir / "metrics.json")
    print(f"test: accuracy {metrics['accuracy']:.2%}, macro-F1 {metrics['macro_f1']:.2%}"
          + "".join(f" | {b}: {v['macro_f1']:.2%}" for b, v in by_length.items() if b != "full")
          + f" -> {rel(model_dir / 'metrics.json')}", flush=True)
    return metrics


def main(argv=None) -> dict:
    setup_console()
    parser = argparse.ArgumentParser(description="Đánh giá model.joblib trên tập test, ghi metrics.json")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args(argv)
    return evaluate(args.model_dir, args.data_dir)


if __name__ == "__main__":
    main()
