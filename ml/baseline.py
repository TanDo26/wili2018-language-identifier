"""Baseline: luôn đoán ngôn ngữ có nhiều đoạn nhất trong train. Là mốc thấp nhất để so sánh."""
from sklearn.dummy import DummyClassifier

PARAM = None  # không có tham số để tune
DEFAULT = None
GRID = ()


def build_model(value=DEFAULT):
    return DummyClassifier(strategy="most_frequent")
