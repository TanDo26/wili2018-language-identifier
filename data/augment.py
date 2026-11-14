"""Tăng cường dữ liệu cho văn bản ngắn và tạo tập đánh giá ngắn -> data/processed/.

    python -m data.augment

WiLI chỉ có đoạn văn từ 140 ký tự trở lên, trong khi người dùng có thể nhập từ 20 ký tự;
mô hình chỉ học trên đoạn dài đoán sai nhiều với input ngắn (experiments/results/).

- train_crops.csv : mỗi đoạn train sinh CROPS_PER_TEXT đoạn cắt ngắn, độ dài ngẫu nhiên
                    CROP_MIN–CROP_MAX ký tự. Chỉ cắt từ train SAU khi đã chia tập, nên các
                    mảnh không bao giờ lẫn sang val/test.
- val_short.csv,
  test_short.csv  : mỗi đoạn val/test cắt một lần ở mỗi độ dài trong SHORT_LENGTHS –
                    dùng để chọn cấu hình (val) và báo cáo kết quả theo độ dài (test).

Cột: id, text, label, source_id (+ length ở tập đánh giá). Seed cố định nên chạy lại ra
đúng cùng nội dung.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data.common import (
    PROCESSED_DIR,
    SEED,
    SHORT_LENGTHS,
    TEST_SHORT_PATH,
    TRAIN_CROPS_PATH,
    VAL_SHORT_PATH,
    load_split,
    rel,
    setup_console,
    write_split,
)

CROPS_PER_TEXT = 2
CROP_MIN, CROP_MAX = 20, 140
# Từ quá dài (tiếng ghép, chữ không dùng dấu cách lẫn vài ký tự Latin...) thì cắt ngang từ.
MAX_WORD_OVERRUN = 20


def crop(text: str, length: int, rng: np.random.Generator) -> str:
    """Cắt một đoạn dài ít nhất `length` ký tự tại vị trí ngẫu nhiên.

    Văn bản có dấu cách thì đoạn cắt bắt đầu và kết thúc ở ranh giới từ (giống người dùng
    gõ vài từ); chữ không dùng dấu cách (Trung, Nhật, Thái...) thì cắt đúng `length` ký tự.
    """
    if len(text) <= length:
        return text
    start = int(rng.integers(0, len(text) - length + 1))
    if " " not in text:
        return text[start:start + length]
    start = text.rfind(" ", 0, start) + 1
    end = text.find(" ", start + length)
    if end == -1:
        end = len(text)
    if end - start > length + MAX_WORD_OVERRUN:
        end = start + length
        if text[end - 1] == " ":  # đã gộp khoảng trắng nên chỉ cần lùi qua một dấu cách
            end += 1
    return text[start:end].strip()


def make_train_crops(train: pd.DataFrame, n_per_text: int = CROPS_PER_TEXT,
                     min_len: int = CROP_MIN, max_len: int = CROP_MAX,
                     seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for r in train.sort_values("id").itertuples(index=False):
        for k in range(n_per_text):
            piece = crop(r.text, int(rng.integers(min_len, max_len + 1)), rng)
            if piece != r.text:
                rows.append((f"{r.id}#c{k}", piece, r.label, r.id))
    return pd.DataFrame(rows, columns=["id", "text", "label", "source_id"])


def make_short_eval(df: pd.DataFrame, lengths=SHORT_LENGTHS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for length in lengths:
        for r in df.sort_values("id").itertuples(index=False):
            rows.append((f"{r.id}@{length}", crop(r.text, length, rng), r.label, r.id, length))
    return pd.DataFrame(rows, columns=["id", "text", "label", "source_id", "length"])


def main(argv=None):
    setup_console()
    parser = argparse.ArgumentParser(description="Tạo train_crops và tập đánh giá ngắn")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    outputs = {
        TRAIN_CROPS_PATH.name: make_train_crops(load_split("train", args.processed_dir), seed=args.seed),
        VAL_SHORT_PATH.name: make_short_eval(load_split("val", args.processed_dir), seed=args.seed),
        TEST_SHORT_PATH.name: make_short_eval(load_split("test", args.processed_dir), seed=args.seed),
    }
    for name, df in outputs.items():
        path = args.processed_dir / name
        write_split(df, path)
        print(f"  {name:<16} {len(df):>7} mẫu, độ dài trung vị {df['text'].str.len().median():.0f} "
              f"ký tự -> {rel(path)}")


if __name__ == "__main__":
    main()
