"""So sánh các cấu hình đặc trưng (ml/features.py), chọn trên tập val kể cả val cắt ngắn.

    python -m experiments.feature_search                          # mọi preset, Naive Bayes, không crop
    python -m experiments.feature_search --presets char_wb_1_3_100k --classifiers nb sgd --augment both
    python -m experiments.feature_search --classifiers ml.naive_bayes:build_model   # mô hình của B
    python -m experiments.feature_search --final --presets char_wb_1_3_100k --classifiers nb --augment no
        -> train cấu hình đã chọn, dự đoán trên test + test_short cho experiments.error_analysis

Bộ phân loại "nb", "sgd" chỉ là mô hình thăm dò cố định để so sánh đặc trưng, không thay cho
mô hình của B; khi B có build_model() thì truyền "module:hàm" vào --classifiers.

Mỗi lần chạy được gộp vào experiments/results/feature_search.json (khóa preset|classifier|augment);
bảng feature_search.md được viết lại từ toàn bộ kết quả. Dự đoán chi tiết của --final nằm ở
experiments/output/ (không commit vì nặng).
"""
from __future__ import annotations

import argparse
import importlib
import time

import numpy as np
import pandas as pd
from scipy.sparse import vstack
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB

from data.common import (
    ROOT,
    SEED,
    SHORT_LENGTHS,
    TEST_SHORT_PATH,
    TRAIN_CROPS_PATH,
    VAL_SHORT_PATH,
    load_split,
    read_json,
    read_split,
    rel,
    setup_console,
    write_json,
    write_split,
)
from ml.features import FEATURE_PRESETS, build_vectorizer, preset_params

RESULTS_DIR = ROOT / "experiments" / "results"
OUTPUT_DIR = ROOT / "experiments" / "output"
RESULTS_JSON = RESULTS_DIR / "feature_search.json"
BUCKETS = [str(n) for n in SHORT_LENGTHS] + ["full"]

PROBES = {
    "nb": lambda: MultinomialNB(alpha=0.01),
    "sgd": lambda: SGDClassifier(loss="modified_huber", alpha=1e-6, max_iter=15, tol=None,
                                 random_state=SEED, n_jobs=-1),
}


def make_classifier(spec: str):
    if spec in PROBES:
        return PROBES[spec]()
    module, _, func = spec.partition(":")
    if not func:
        raise SystemExit(f"--classifiers nhận {sorted(PROBES)} hoặc 'module:hàm', không phải {spec!r}")
    return getattr(importlib.import_module(module), func)()


def classifier_mb(clf) -> float:
    """Dung lượng các mảng numpy trong bộ phân loại (xấp xỉ kích thước model.joblib)."""
    return round(sum(v.nbytes for v in vars(clf).values() if isinstance(v, np.ndarray)) / 1e6, 1)


def eval_frame(full: pd.DataFrame, short: pd.DataFrame) -> pd.DataFrame:
    """Gộp tập đầy đủ và tập cắt ngắn, cột bucket = "20" | "30" | ... | "full"."""
    full = full.assign(source_id=full["id"], bucket="full")
    short = short.assign(bucket=short["length"].astype(str))
    return pd.concat([full, short], ignore_index=True)[["id", "source_id", "text", "label", "bucket"]]


def predict_in_chunks(vec, clf, texts: pd.Series, chunk: int = 50_000):
    """Dự đoán theo khối để ma trận thưa của ~500k văn bản test không chiếm hết RAM.

    Trả về (nhãn dự đoán, độ tin cậy cao nhất hoặc None nếu bộ phân loại không có predict_proba).
    """
    preds, confs = [], []
    for start in range(0, len(texts), chunk):
        X = vec.transform(texts.iloc[start:start + chunk])
        preds.append(clf.predict(X))
        if hasattr(clf, "predict_proba"):
            confs.append(clf.predict_proba(X).max(axis=1))
    return np.concatenate(preds), (np.concatenate(confs) if confs else None)


def scores_by_bucket(df: pd.DataFrame, pred) -> dict:
    out = {}
    for bucket in BUCKETS:
        mask = (df["bucket"] == bucket).to_numpy()
        out[bucket] = {
            "accuracy": round(float(accuracy_score(df["label"][mask], pred[mask])), 4),
            "macro_f1": round(float(f1_score(df["label"][mask], pred[mask], average="macro")), 4),
        }
    return out


def fit(preset: str, classifier: str, augment: bool):
    """Fit vectorizer trên đoạn gốc; bộ phân loại học thêm đoạn cắt ngắn nếu augment."""
    train = load_split("train")
    vec = build_vectorizer(preset)
    X = vec.fit_transform(train["text"])
    y = train["label"].to_numpy()
    if augment:
        crops = read_split(TRAIN_CROPS_PATH)
        X = vstack([X, vec.transform(crops["text"])]).tocsr()
        y = np.concatenate([y, crops["label"].to_numpy()])
    clf = make_classifier(classifier)
    clf.fit(X, y)
    return vec, clf, X.shape[0]


def run(preset: str, classifier: str, augment: bool, final: bool) -> dict:
    t0 = time.time()
    vec, clf, n_train = fit(preset, classifier, augment)
    fit_seconds = time.time() - t0

    val = eval_frame(load_split("val"), read_split(VAL_SHORT_PATH))
    val_scores = scores_by_bucket(val, predict_in_chunks(vec, clf, val["text"])[0])
    params = preset_params(preset)
    record = {
        "preset": preset,
        "classifier": classifier,
        "augment": augment,
        "analyzer": params["analyzer"],
        "ngram_range": list(params["ngram_range"]),
        "max_features": params["max_features"],
        "lowercase": params["lowercase"],
        "n_train": n_train,
        "n_features": len(vec.vocabulary_),
        "model_mb": classifier_mb(clf),
        "fit_seconds": round(fit_seconds),
        "val": val_scores,
        "val_score": round(float(np.mean([val_scores[b]["macro_f1"] for b in BUCKETS])), 4),
    }

    if final:
        test = eval_frame(load_split("test"), read_split(TEST_SHORT_PATH))
        pred, conf = predict_in_chunks(vec, clf, test["text"])
        record["test"] = scores_by_bucket(test, pred)
        out = test[["id", "source_id", "label", "bucket"]].assign(
            n_chars=test["text"].str.len(), pred=pred)
        if conf is not None:
            out["confidence"] = conf.round(4)
        path = OUTPUT_DIR / f"predictions_{key(record).replace('|', '__')}.csv"  # Windows cấm "|" trong tên file
        write_split(out, path)
        record["predictions_file"] = rel(path)

    record["seconds_total"] = round(time.time() - t0)
    return record


def key(record: dict) -> str:
    return f"{record['preset']}|{record['classifier'].replace(':', '.')}|{'crop' if record['augment'] else 'nocrop'}"


def to_markdown(results: dict) -> str:
    rows = sorted(results.values(), key=lambda r: -r["val_score"])
    lines = [
        "# So sánh cấu hình đặc trưng",
        "",
        "Đánh giá trên tập **val** (11.725 đoạn) và val cắt ngắn ở 20/30/50/100 ký tự. Mỗi ô là "
        "macro-F1; `điểm` = trung bình 5 cột, dùng để xếp hạng vì người dùng có thể nhập văn bản "
        "ngắn. Bộ phân loại là mô hình thăm dò cố định để so sánh đặc trưng (`nb` = MultinomialNB "
        "alpha 0.01, `sgd` = SGDClassifier modified_huber), không phải mô hình cuối của B.",
        "",
        "| Preset | Phân loại | Crop | " + " | ".join(f"{b} ký tự" if b != "full" else "Đầy đủ" for b in BUCKETS)
        + " | Điểm | Đặc trưng | Mô hình (MB) | Thời gian (s) |",
        "|---|---|---|" + "---:|" * (len(BUCKETS) + 4),
    ]
    for r in rows:
        cells = " | ".join(f"{r['val'][b]['macro_f1']:.1%}" for b in BUCKETS)
        lines.append(f"| `{r['preset']}` | {r['classifier']} | {'có' if r['augment'] else 'không'} | {cells} "
                     f"| **{r['val_score']:.1%}** | {r['n_features']:,} | {r['model_mb']} | {r['fit_seconds']} |")

    finals = [r for r in rows if "test" in r]
    if finals:
        lines += ["", "## Kết quả trên tập test (chỉ cấu hình đã chọn)", "",
                  "| Preset | Phân loại | Crop | " + " | ".join(
                      f"{b} ký tự" if b != "full" else "Đầy đủ" for b in BUCKETS) + " |",
                  "|---|---|---|" + "---:|" * len(BUCKETS)]
        for r in finals:
            cells = " | ".join(f"{r['test'][b]['accuracy']:.1%} / {r['test'][b]['macro_f1']:.1%}" for b in BUCKETS)
            lines.append(f"| `{r['preset']}` | {r['classifier']} | {'có' if r['augment'] else 'không'} | {cells} |")
        lines += ["", "Mỗi ô: accuracy / macro-F1. Phân tích lỗi chi tiết: [error_analysis.md](error_analysis.md)."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    setup_console()
    parser = argparse.ArgumentParser(description="So sánh cấu hình đặc trưng trên tập val")
    parser.add_argument("--presets", nargs="+", default=list(FEATURE_PRESETS))
    parser.add_argument("--classifiers", nargs="+", default=["nb"])
    parser.add_argument("--augment", choices=("no", "yes", "both"), default="no")
    parser.add_argument("--final", action="store_true", help="thêm đánh giá trên test và lưu dự đoán")
    args = parser.parse_args(argv)

    if not VAL_SHORT_PATH.exists():
        raise SystemExit("Chưa có val_short.csv – chạy `python -m data.augment` trước.")
    augments = {"no": [False], "yes": [True], "both": [False, True]}[args.augment]
    results = read_json(RESULTS_JSON) if RESULTS_JSON.exists() else {}

    for preset in args.presets:
        for classifier in args.classifiers:
            for augment in augments:
                record = run(preset, classifier, augment, args.final)
                results[key(record)] = record
                write_json(results, RESULTS_JSON)  # ghi sau mỗi lần chạy để không mất kết quả
                print(f"{key(record):<40} điểm val {record['val_score']:.1%}  "
                      + "  ".join(f"{b}:{record['val'][b]['macro_f1']:.1%}" for b in BUCKETS)
                      + f"  ({record['seconds_total']}s)", flush=True)

    (RESULTS_DIR / "feature_search.md").write_text(to_markdown(results), encoding="utf-8")
    print(f"Đã ghi {rel(RESULTS_JSON)} và {rel(RESULTS_DIR / 'feature_search.md')}")


if __name__ == "__main__":
    main()
