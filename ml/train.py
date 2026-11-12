"""Train một mô hình trên split của nhóm, lưu sklearn Pipeline (vectorizer + bộ phân loại).

    make retrain ARGS="--algorithm svm"     # cách thường dùng: scripts/retrain.py gọi file này, rồi ml.evaluate
    python -m ml.train --out-dir experiments/output/ml/nb --algorithm naive_bayes --augment --tune

Tham số theo thỏa thuận với scripts/retrain.py: --data-dir, --out-dir, --features, --algorithm,
--augment, --feedback. Thêm cho thí nghiệm: --tune (thử GRID của mô hình) và --value (đặt tham số).

Ghi vào --out-dir:
  model.joblib     Pipeline có predict / predict_proba trên văn bản đã chuẩn hóa (API nạp trực tiếp)
  train_info.json  thuật toán, tham số, kết quả tune, macro-F1 trên val, số mẫu, thời gian

Vectorizer chỉ học từ vựng trên train gốc; bộ phân loại học thêm các đoạn cắt ngắn (--augment) và báo cáo
đã duyệt (--feedback). Tham số được chọn trên val theo trung bình macro-F1 của val đầy đủ và val cắt
ngắn 20/30/50/100 ký tự (giống experiments/feature_search.py). Tập test chỉ dùng trong ml.evaluate.
"""
import argparse
import importlib
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import vstack
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from data.common import (
    FEEDBACK_TRAIN_PATH,
    PROCESSED_DIR,
    TRAIN_CROPS_PATH,
    VAL_SHORT_PATH,
    load_split,
    read_split,
    rel,
    setup_console,
    write_json,
)
from ml.features import DEFAULT_PRESET, FEATURE_PRESETS, build_vectorizer

ALGORITHMS = ("baseline", "naive_bayes", "svm")  # tên module trong ml/, trùng scripts/retrain.py


def algorithm_module(name: str):
    return importlib.import_module(f"ml.{name}")


def with_buckets(full: pd.DataFrame, short: pd.DataFrame) -> pd.DataFrame:
    """Gộp tập đầy đủ và tập cắt ngắn; cột bucket = "20" | "30" | "50" | "100" | "full"."""
    return pd.concat([full.assign(bucket="full"), short.assign(bucket=short["length"].astype(str))],
                     ignore_index=True)


def bucket_order(buckets) -> list[str]:
    return sorted(set(buckets), key=lambda b: (b == "full", int(b) if b.isdigit() else 0))


def scores_by_bucket(df: pd.DataFrame, pred) -> dict:
    """{bucket: {accuracy, macro_f1, n}} và "score" = trung bình macro-F1 các bucket."""
    out = {}
    for bucket in bucket_order(df["bucket"]):
        mask = (df["bucket"] == bucket).to_numpy()
        y = df["label"].to_numpy()[mask]
        out[bucket] = {"accuracy": round(float(accuracy_score(y, pred[mask])), 4),
                       "macro_f1": round(float(f1_score(y, pred[mask], average="macro")), 4),
                       "n": int(mask.sum())}
    out["score"] = round(float(np.mean([v["macro_f1"] for v in out.values()])), 4)
    return out


def slim(clf) -> None:
    """Thu nhỏ và tăng tốc model.joblib mà không đổi kết quả dự đoán.

    - bỏ feature_count_ của MultinomialNB (chỉ cần cho partial_fit), lưu trọng số float32;
    - lưu trọng số (lớp x đặc trưng) theo thứ tự Fortran: dự đoán nhân X với W.T, nếu W.T không liền theo
      thứ tự C thì scipy chép cả ma trận (~94 MB) ở mỗi lần gọi; đo trên NB 100k đặc trưng: 41 ms -> 1,5 ms.
    """
    if hasattr(clf, "feature_count_"):
        del clf.feature_count_
    for name in ("feature_log_prob_", "coef_"):
        if hasattr(clf, name):
            setattr(clf, name, np.asfortranarray(getattr(clf, name), dtype=np.float32))


def train(args) -> Path:
    module = algorithm_module(args.algorithm)
    t0 = time.time()

    train_df = load_split("train", args.data_dir)
    vectorizer = build_vectorizer(args.features)
    blocks = [vectorizer.fit_transform(train_df["text"])]
    labels = [train_df["label"].to_numpy()]
    extra_files = ([args.data_dir / TRAIN_CROPS_PATH.name] if args.augment else []) + \
                  ([args.data_dir / FEEDBACK_TRAIN_PATH.name] if args.feedback else [])
    for path in extra_files:
        if not path.exists():
            raise SystemExit(f"Không thấy {rel(path)}: chạy `python -m data.pipeline` (hoặc data.feedback) trước.")
        extra = read_split(path)
        blocks.append(vectorizer.transform(extra["text"]))
        labels.append(extra["label"].to_numpy())
    X = vstack(blocks).tocsr()
    y = np.concatenate(labels)

    val_short_path = args.data_dir / VAL_SHORT_PATH.name
    if not val_short_path.exists():
        raise SystemExit(f"Không thấy {rel(val_short_path)}: chạy `python -m data.pipeline` trước.")
    val = with_buckets(load_split("val", args.data_dir), read_split(val_short_path))
    X_val = vectorizer.transform(val["text"])

    tuning = []
    value = module.DEFAULT if args.value is None else args.value
    if args.tune and module.GRID:
        for candidate in module.GRID:
            scores = scores_by_bucket(val, module.build_model(candidate).fit(X, y).predict(X_val))
            tuning.append({"value": candidate, **scores})
            print(f"{args.algorithm} {module.PARAM}={candidate}: điểm val {scores['score']:.4f}", flush=True)
        value = max(tuning, key=lambda t: t["score"])["value"]

    t_fit = time.time()
    clf = module.build_model(value)
    clf.fit(X, y)
    slim(clf)
    if hasattr(module, "calibrate"):
        clf = module.calibrate(clf, X_val, val["label"].to_numpy())
    fit_seconds = time.time() - t_fit

    model = Pipeline([("features", vectorizer), ("classifier", clf)])
    val_scores = scores_by_bucket(val, clf.predict(X_val))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out_dir / "model.joblib", compress=3)
    write_json({
        "algorithm": args.algorithm,
        "classifier": type(clf).__name__,
        "hyperparameter": module.PARAM,
        "value": value,
        "features": args.features,
        "augment": args.augment,
        "feedback": args.feedback,
        "n_train": int(X.shape[0]),
        "n_features": len(vectorizer.vocabulary_),
        "n_classes": len(clf.classes_),
        "tuning": tuning,
        "val": val_scores,
        "fit_seconds": round(fit_seconds, 1),
        "seconds_total": round(time.time() - t0, 1),
    }, args.out_dir / "train_info.json")
    print(f"{args.algorithm} ({module.PARAM}={value}): val đầy đủ macro-F1 {val_scores['full']['macro_f1']:.2%}, "
          f"điểm val {val_scores['score']:.2%} -> {rel(args.out_dir)}", flush=True)
    return args.out_dir


def main(argv=None) -> Path:
    setup_console()
    parser = argparse.ArgumentParser(description="Train mô hình nhận diện ngôn ngữ trên split của nhóm")
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--features", choices=sorted(FEATURE_PRESETS), default=DEFAULT_PRESET)
    parser.add_argument("--algorithm", choices=ALGORITHMS, default="naive_bayes")
    parser.add_argument("--augment", action="store_true", help="học thêm train_crops.csv (văn bản ngắn)")
    parser.add_argument("--feedback", action="store_true", help="học thêm train_feedback.csv (báo cáo đã duyệt)")
    parser.add_argument("--tune", action="store_true", help="thử GRID của mô hình, chọn theo điểm val")
    parser.add_argument("--value", type=float, help="đặt tham số (alpha của NB, C của SVM)")
    return train(parser.parse_args(argv))


if __name__ == "__main__":
    main()
