"""The model-calling contract (used by api/).
    from predict import predict
    predict(["Con sông chảy lặng lẽ qua phố cổ."], top_k=3, model_dir="models/v1")
returns one dict per text:
    {"label": "vie", "iso": "vie", "confidence": 0.93,
     "candidates": [{"label","iso","confidence"}, ...],   # length top_k, best first
     "model_version": "v1"}
Note: SVM confidences are softmax(decision_function): a ranking aid, NOT calibrated probabilities."""
import json
from functools import lru_cache
from pathlib import Path
import joblib, numpy as np

DEFAULT_DIR = "models/v1"

@lru_cache(maxsize=8)
def load(model_dir=DEFAULT_DIR):
    d = Path(model_dir)
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    return joblib.load(d / "vectorizer.joblib"), joblib.load(d / "model.joblib"), cfg

def _scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    z = model.decision_function(X); z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z); return e / e.sum(axis=1, keepdims=True)

def predict(texts, top_k=3, model_dir=DEFAULT_DIR):
    if isinstance(texts, str):
        texts = [texts]
    vec, model, cfg = load(str(model_dir))
    P = _scores(model, vec.transform(texts)); iso = cfg.get("iso_codes", {}); out = []
    for row in P:
        cands = []
        for i in np.argsort(-row)[:top_k]:
            lab = str(model.classes_[i])
            cands.append({"label": lab, "iso": iso.get(lab, lab), "confidence": float(row[i])})
        out.append({**cands[0], "candidates": cands, "model_version": cfg["version"]})
    return out
