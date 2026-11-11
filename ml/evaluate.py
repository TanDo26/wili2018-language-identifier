"""Accuracy, macro-F1, confusion matrix, error analysis, latency, mixed-language check.
  python evaluate.py models/v1 --split val
  python evaluate.py models/v1 --split test     # allowed ONCE per model folder"""
import argparse, json, sys, time, joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from data import load_splits
from predict import predict

BUCKETS = [(0, 50), (50, 100), (100, 200), (200, 500), (500, 10**9)]

def mixed_check(model_dir, X, y, n=500, seed=42):
    rng = np.random.default_rng(seed); hits = done = 0
    for _ in range(n):
        i, j = rng.integers(len(X), size=2)
        if y[i] == y[j]:
            continue
        got = predict([X[i] + " " + X[j]], top_k=1, model_dir=model_dir)[0]["label"]
        hits += got in (y[i], y[j]); done += 1
    return {"pairs": done, "prediction_is_one_of_the_two": hits / max(done, 1)}

def evaluate(model_dir, split, data):
    d = Path(model_dir); flag = d / ".test_done"
    if split == "test" and flag.exists():
        sys.exit("Test set already used for this model. Do not re-run it.")
    X, y = data[split]; y = np.asarray(y)
    vec = joblib.load(d / "vectorizer.joblib"); model = joblib.load(d / "model.joblib")
    pred = np.asarray(model.predict(vec.transform(X)))
    labels = list(model.classes_)
    cm = confusion_matrix(y, pred, labels=labels); np.save(d / f"confusion_{split}.npy", cm)
    off = cm.copy(); np.fill_diagonal(off, 0)
    top = np.dstack(np.unravel_index(np.argsort(-off, axis=None)[:15], off.shape))[0]
    pairs = [{"true": labels[i], "pred": labels[j], "count": int(off[i, j])} for i, j in top if off[i, j]]
    lens = np.array([len(t) for t in X]); by_len = {}
    for lo, hi in BUCKETS:
        m = (lens >= lo) & (lens < hi)
        if m.any():
            by_len[f"{lo}-{hi if hi < 10**9 else 'inf'}"] = {"n": int(m.sum()), "accuracy": float((pred[m] == y[m]).mean())}
    times = []
    for t in X[:200]:
        s = time.perf_counter(); predict([t], top_k=3, model_dir=d); times.append((time.perf_counter() - s) * 1000)
    res = {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"),
           "top_confused_pairs": pairs, "accuracy_by_text_length": by_len,
           "mixed_language_check": mixed_check(d, X, list(y)),
           "latency_ms_median": float(np.median(times)), "latency_ms_p95": float(np.percentile(times, 95))}
    mp = d / "metrics.json"; metrics = json.loads(mp.read_text())
    metrics[split] = res; mp.write_text(json.dumps(metrics, indent=2))
    if split == "test":
        flag.touch()
    return res

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("model_dir")
    p.add_argument("--split", choices=["val", "test"], default="val"); a = p.parse_args()
    print(json.dumps(evaluate(a.model_dir, a.split, load_splits()), indent=2))
