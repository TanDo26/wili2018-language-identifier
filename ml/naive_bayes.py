"""Multinomial Naive Bayes trên TF-IDF n-gram ký tự (ml/features.py)."""
from sklearn.naive_bayes import MultinomialNB

PARAM = "alpha"
DEFAULT = 0.003  # chọn bằng `python -m ml.train --algorithm naive_bayes --augment --tune` (xem ml/README.md)
GRID = (0.001, 0.003, 0.01, 0.03, 0.1)


def build_model(value=DEFAULT):
    return MultinomialNB(alpha=value)
