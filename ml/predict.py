"""Gọi mô hình từ Python, dùng cho thí nghiệm và demo:

    python -m ml.predict models/v1 "Con sông chảy lặng lẽ qua phố cổ."

API (api/main.py) không import file này mà làm đúng các bước giống hệt: chuẩn hóa bằng
data.common.normalize_text, rồi gọi predict_proba của Pipeline trong models/vN/model.joblib.
"""
import argparse
import json
from functools import lru_cache
from pathlib import Path

import joblib

from data.common import normalize_text, setup_console


@lru_cache(maxsize=2)
def load(model_dir: str):
    return joblib.load(Path(model_dir) / "model.joblib")


def predict(texts, model_dir, top_k: int = 3) -> list[dict]:
    """Mỗi văn bản -> {"label", "confidence", "candidates": [{"label", "confidence"}, ...]} (tốt nhất trước)."""
    if isinstance(texts, str):
        texts = [texts]
    model = load(str(model_dir))
    probabilities = model.predict_proba([normalize_text(t) for t in texts])
    results = []
    for row in probabilities:
        candidates = [{"label": str(model.classes_[i]), "confidence": round(float(row[i]), 4)}
                      for i in row.argsort()[::-1][:top_k]]
        results.append({**candidates[0], "candidates": candidates})
    return results


def main(argv=None):
    setup_console()
    parser = argparse.ArgumentParser(description="Dự đoán ngôn ngữ bằng một version mô hình")
    parser.add_argument("model_dir")
    parser.add_argument("texts", nargs="+")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args(argv)
    print(json.dumps(predict(args.texts, args.model_dir, args.top_k), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
