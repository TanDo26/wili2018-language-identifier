"""Đặc trưng TF-IDF trên n-gram ký tự cho bài toán nhận diện ngôn ngữ (phần của A).

Cách dùng trong train.py:

    from ml.features import build_vectorizer
    vec = build_vectorizer()                      # cấu hình mặc định DEFAULT_PRESET
    vec = build_vectorizer("char_wb_1_4_100k")    # hoặc một preset khác
    vec = build_vectorizer(max_features=50_000)   # ghi đè từng tham số

Các preset được so sánh bằng `python -m experiments.feature_search` (chọn trên tập val,
kể cả val cắt ngắn); kết quả ở experiments/results/feature_search.md.

Lưu ý khi dùng:
- Fit vectorizer trên đoạn văn gốc (train.csv); các đoạn cắt ngắn trong train_crops.csv
  chỉ cần transform rồi đưa vào bộ phân loại.
- Vectorizer chỉ gồm thành phần sklearn chuẩn (không có hàm tự viết) nên model.joblib
  load được ở API mà không cần copy thư mục ml/ hay data/. Đổi lại, API phải chuẩn hóa
  input giống data.common.normalize_text (Unicode NFC + gộp khoảng trắng) trước khi predict.
- strip_accents luôn tắt: dấu là tín hiệu quan trọng (vd. tiếng Việt, Séc, Rumani).
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

COMMON = {"lowercase": True, "strip_accents": None, "sublinear_tf": True, "dtype": np.float32}

FEATURE_PRESETS = {
    # char_wb: n-gram chỉ lấy trong từng từ (có đệm dấu cách ở hai đầu từ)
    "char_wb_1_3_100k": {"analyzer": "char_wb", "ngram_range": (1, 3), "max_features": 100_000},
    "char_wb_1_3_300k": {"analyzer": "char_wb", "ngram_range": (1, 3), "max_features": 300_000},
    "char_wb_1_4_100k": {"analyzer": "char_wb", "ngram_range": (1, 4), "max_features": 100_000},
    "char_wb_1_5_100k": {"analyzer": "char_wb", "ngram_range": (1, 5), "max_features": 100_000},
    "char_wb_1_3_100k_cased": {"analyzer": "char_wb", "ngram_range": (1, 3), "max_features": 100_000,
                               "lowercase": False},
    "char_wb_1_4_100k_cased": {"analyzer": "char_wb", "ngram_range": (1, 4), "max_features": 100_000,
                               "lowercase": False},
    # char: n-gram cắt ngang cả ranh giới từ
    "char_1_3_100k": {"analyzer": "char", "ngram_range": (1, 3), "max_features": 100_000},
}

# Chọn theo macro-F1 trung bình trên val (đầy đủ + cắt ngắn 20/30/50/100 ký tự); 1–5-gram và
# bản giữ hoa/thường không tốt hơn, 300k đặc trưng không tốt hơn 100k mà mô hình nặng gấp 3.
DEFAULT_PRESET = "char_wb_1_4_100k"


def preset_params(preset: str | dict = DEFAULT_PRESET, **overrides) -> dict:
    if isinstance(preset, str):
        if preset not in FEATURE_PRESETS:
            raise ValueError(f"preset không tồn tại: {preset!r}; có: {sorted(FEATURE_PRESETS)}")
        preset = FEATURE_PRESETS[preset]
    return {**COMMON, **preset, **overrides}


def build_vectorizer(preset: str | dict = DEFAULT_PRESET, **overrides) -> TfidfVectorizer:
    return TfidfVectorizer(**preset_params(preset, **overrides))
