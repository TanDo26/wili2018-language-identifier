"""Hằng số, đường dẫn và hàm dùng chung cho pipeline dữ liệu WiLI-2018."""
from __future__ import annotations

import functools
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
WILI_DIR = RAW_DIR / "wili-2018"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
REPORTS_DIR = DATA_DIR / "reports"

SPLIT_IDS_PATH = SPLITS_DIR / "split_ids.csv"
CLEAN_REPORT_PATH = INTERIM_DIR / "clean_report.json"
LEAKAGE_REPORT_PATH = PROCESSED_DIR / "leakage_report.json"
LABELS_PATH = PROCESSED_DIR / "labels.csv"
STATS_PATH = PROCESSED_DIR / "stats.json"
TRAIN_CROPS_PATH = PROCESSED_DIR / "train_crops.csv"
VAL_SHORT_PATH = PROCESSED_DIR / "val_short.csv"
TEST_SHORT_PATH = PROCESSED_DIR / "test_short.csv"
FEEDBACK_TRAIN_PATH = PROCESSED_DIR / "train_feedback.csv"

SEED = 42
# API nhận văn bản từ 20 ký tự, nhưng WiLI không có đoạn nào ngắn hơn 140 ký tự,
# nên có thêm các tập đánh giá cắt ngắn ở những độ dài này (data/augment.py).
MIN_TEXT_LEN = 20
SHORT_LENGTHS = (20, 30, 50, 100)
VAL_PER_LANG = 50
# Ngôn ngữ có ít mẫu (vd. che chỉ còn ~60 đoạn sau khử trùng lặp) thì val lấy tối đa
# một nửa, để train vẫn còn dữ liệu.
VAL_MAX_FRACTION = 0.5
SPLITS = ("train", "val", "test")

# Tiền tố id cho biết dòng đến từ file gốc nào của WiLI (x_train.txt / x_test.txt),
# số phía sau là chỉ số dòng (0-based) trong file đó.
ID_PREFIX = {"train": "tr", "test": "te"}

_WS_RE = re.compile(r"\s+")
# Ký tự điều khiển (Unicode Cc) và BOM; khoảng trắng đã được đổi thành " " trước khi lọc.
_CTRL_RE = re.compile("[\x00-\x1f\x7f-\x9f﻿]")


def setup_console() -> None:
    """Console Windows mặc định dùng cp1252, in tiếng Việt sẽ lỗi -> chuyển sang UTF-8."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def normalize_text(text: str) -> str:
    """Chuẩn hóa Unicode NFC, bỏ ký tự điều khiển / BOM, gộp khoảng trắng."""
    text = unicodedata.normalize("NFC", text)
    text = _CTRL_RE.sub("", _WS_RE.sub(" ", text))
    return _WS_RE.sub(" ", text).strip()


@functools.lru_cache(maxsize=None)
def _is_letter_or_mark(ch: str) -> bool:
    return unicodedata.category(ch)[0] in "LM"


def near_dup_key(text: str) -> str:
    """Khóa so khớp "gần trùng": chỉ giữ chữ cái và dấu (L*, M*) sau casefold.

    Hai đoạn chỉ khác nhau ở chữ số, dấu câu, khoảng trắng hay hoa/thường
    sẽ có cùng khóa.
    """
    text = unicodedata.normalize("NFC", text).casefold()
    drop = {ord(ch): None for ch in set(text) if not _is_letter_or_mark(ch)}
    return text.translate(drop)


def make_ids(source: str, n: int) -> list[str]:
    return [f"{ID_PREFIX[source]}_{i:06d}" for i in range(n)]


def read_lines(path: Path) -> list[str]:
    """Đọc file văn bản, mỗi dòng một mẫu.

    Chỉ tách theo "\\n" (không dùng splitlines) để các ký tự như U+2028
    nằm trong đoạn văn không làm lệch thứ tự dòng so với file nhãn.
    """
    with open(path, encoding="utf-8", newline="\n") as f:
        content = f.read()
    if content.endswith("\n"):
        content = content[:-1]
    return [line.rstrip("\r") for line in content.split("\n")]


def read_split(path: Path) -> pd.DataFrame:
    """Đọc CSV (id, text, label) – giữ nguyên chuỗi, không đổi "nan"/"NA" thành NaN.

    Lưu ý: WiLI có nhãn "nan" (tiếng Mân Nam), nên không được để pandas tự suy NaN.
    """
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)


def load_split(split: str, processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Đọc tập đã chia: load_split("train") -> DataFrame cột id, text, label."""
    if split not in SPLITS:
        raise ValueError(f"split phải là một trong {SPLITS}, nhận được {split!r}")
    return read_split(processed_dir / f"{split}.csv")


def write_split(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def file_hash(path: Path, algo: str = "sha256", chunk_size: int = 1 << 20) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def length_stats(lengths) -> dict:
    """Thống kê mô tả độ dài văn bản (số ký tự)."""
    arr = np.asarray(lengths, dtype=float)
    if arr.size == 0:
        return {"count": 0}
    p = [round(float(v), 2) for v in np.percentile(arr, [5, 25, 50, 75, 95])]
    return {
        "count": int(arr.size),
        "min": int(arr.min()),
        "p5": p[0],
        "p25": p[1],
        "median": p[2],
        "mean": round(float(arr.mean()), 2),
        "p75": p[3],
        "p95": p[4],
        "max": int(arr.max()),
    }


def rel(path: Path) -> str:
    """Đường dẫn tương đối so với gốc repo, dạng POSIX (để ghi vào report)."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()
