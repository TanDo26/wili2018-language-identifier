"""Train majority baseline / Multinomial NB / Linear SVM.
  python train.py --model nb --tune --version v1            # -> models/v1/  (deliverable)
  python train.py --model svm --tune --out experiments/svm  # comparison run
  python train.py --model majority --out experiments/majority
Output folder: vectorizer.joblib, model.joblib, config.json, metrics.json
The test set is never used here."""
import argparse, json, time, joblib
from datetime import datetime, timezone
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from data import load_splits

def slim(vec, clf):
    """Shrink saved files: drop training-only arrays, store weights as float32."""
    if hasattr(vec, "stop_words_"):
        vec.stop_words_ = set()
    if hasattr(clf, "feature_count_"):
        del clf.feature_count_
    for a in ("feature_log_prob_", "coef_"):
        if hasattr(clf, a):
            setattr(clf, a, getattr(clf, a).astype(np.float32))

def make_clf(kind, value):
    return {"majority": lambda: DummyClassifier(strategy="most_frequent"),
            "nb": lambda: MultinomialNB(alpha=value),
            "svm": lambda: LinearSVC(C=value)}[kind]()

def run(data, kind, version, out, value=None, nmax=3, max_features=100_000, tune=False):
    Xtr, ytr = data["train"]; Xva, yva = data["val"]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, nmax), sublinear_tf=True,
                          max_features=10 if kind == "majority" else max_features)
    A = vec.fit_transform(Xtr); V = vec.transform(Xva)
    default = {"nb": 0.1, "svm": 1.0, "majority": None}[kind]
    tuning = []
    if tune and kind != "majority":
        for v in ([0.01, 0.03, 0.1, 0.3, 1.0] if kind == "nb" else [0.1, 0.3, 1.0, 3.0]):
            f1 = f1_score(yva, make_clf(kind, v).fit(A, ytr).predict(V), average="macro")
            tuning.append({"value": v, "val_macro_f1": f1}); print(kind, v, round(f1, 4))
        value = max(tuning, key=lambda t: t["val_macro_f1"])["value"]
    value = default if value is None else value
    t0 = time.time(); clf = make_clf(kind, value).fit(A, ytr); secs = time.time() - t0
    slim(vec, clf)
    pred = clf.predict(V)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    joblib.dump(vec, out / "vectorizer.joblib", compress=3); joblib.dump(clf, out / "model.joblib", compress=3)
    labels = [str(c) for c in clf.classes_]
    cfg = {"version": version, "model": kind,
           "hyperparameter": {"nb": "alpha", "svm": "C"}.get(kind), "value": value,
           "ngram_range": [1, nmax], "labels": labels,
           "iso_codes": {l: l for l in labels},   # replace with a real code map if labels are names
           "trained_on": data["source"], "trained_at": datetime.now(timezone.utc).isoformat()}
    metrics = {"version": version, "model": kind, "trained_on": data["source"],
               "val": {"accuracy": accuracy_score(yva, pred), "macro_f1": f1_score(yva, pred, average="macro")},
               "test": None, "train_seconds": secs, "tuning": tuning}
    (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(kind, metrics["val"], "->", out)
    return out

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["majority", "nb", "svm"], required=True)
    p.add_argument("--version", default="v1")
    p.add_argument("--out", default=None)
    p.add_argument("--value", type=float, default=None, help="alpha (nb) or C (svm)")
    p.add_argument("--ngram-max", type=int, default=3)
    p.add_argument("--max-features", type=int, default=100_000, help="smaller = smaller model file")
    p.add_argument("--tune", action="store_true")
    a = p.parse_args()
    run(load_splits(), a.model, a.version, a.out or f"models/{a.version}",
        a.value, a.ngram_max, a.max_features, tune=a.tune)