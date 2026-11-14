"""Gộp các báo cáo lỗi đã được Data Scientist duyệt vào dữ liệu train (yêu cầu RV-3).

    python -m data.feedback path/to/accepted_reports.csv

File đầu vào (tạo bằng `python -m data.export_reports`), CSV UTF-8 với các cột:
    text       văn bản người dùng đã nhập
    label      verified_language do Data Scientist gán (một trong 235 nhãn WiLI)
    report_id  (tùy chọn)

Quy tắc, để báo cáo của người dùng không làm bẩn hay rò rỉ dữ liệu:
  - chuẩn hóa giống clean.py; bỏ mẫu dưới MIN_TEXT_LEN ký tự hoặc không có chữ cái;
  - nhãn phải có trong data/processed/labels.csv;
  - bỏ trùng trong file, bỏ mẫu gần trùng với val/test (rò rỉ) và với train hiện có (thừa);
  - chỉ ghi ra data/processed/train_feedback.csv – val/test không bao giờ thay đổi.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data.common import (
    FEEDBACK_TRAIN_PATH,
    LABELS_PATH,
    MIN_TEXT_LEN,
    PROCESSED_DIR,
    SPLITS,
    load_split,
    near_dup_key,
    normalize_text,
    read_split,
    rel,
    setup_console,
    write_json,
    write_split,
)

REASONS = ("too_short", "no_letters", "unknown_label", "duplicate", "overlap_val_test", "already_in_train")


def merge_feedback(feedback: pd.DataFrame, splits: dict[str, pd.DataFrame],
                   labels: set[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Trả về (mẫu được nhận, mẫu bị loại kèm lý do, report)."""
    missing = {"text", "label"} - set(feedback.columns)
    if missing:
        raise ValueError(f"file feedback thiếu cột {sorted(missing)}")

    df = feedback.copy()
    if "report_id" not in df.columns:
        df["report_id"] = [str(i) for i in range(len(df))]
    df["id"] = "fb_" + df["report_id"].astype(str)
    df["text"] = df["text"].map(normalize_text)
    df["label"] = df["label"].str.strip()
    df["key"] = df["text"].map(near_dup_key)

    eval_keys = set().union(*(set(splits[s]["text"].map(near_dup_key)) for s in ("val", "test")))
    train_keys = set(splits["train"]["text"].map(near_dup_key))

    df["reason"] = None
    rules = [
        ("too_short", df["text"].str.len() < MIN_TEXT_LEN),
        ("no_letters", df["key"] == ""),
        ("unknown_label", ~df["label"].isin(labels)),
        ("duplicate", df["key"].duplicated(keep="first")),
        ("overlap_val_test", df["key"].isin(eval_keys)),
        ("already_in_train", df["key"].isin(train_keys)),
    ]
    for reason, mask in rules:
        df.loc[mask & df["reason"].isna(), "reason"] = reason

    accepted = df[df["reason"].isna()][["id", "text", "label", "report_id"]].reset_index(drop=True)
    removed = df[df["reason"].notna()][["id", "label", "reason", "text"]].reset_index(drop=True)
    report = {
        "input": len(df),
        "accepted": len(accepted),
        "removed": {r: int((df["reason"] == r).sum()) for r in REASONS},
        "per_label": {k: int(v) for k, v in accepted["label"].value_counts().items()},
    }
    return accepted, removed, report


def main(argv=None):
    setup_console()
    parser = argparse.ArgumentParser(description="Gộp báo cáo lỗi đã duyệt vào train")
    parser.add_argument("feedback_csv", type=Path)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args(argv)

    accepted, removed, report = merge_feedback(
        read_split(args.feedback_csv),
        {name: load_split(name, args.processed_dir) for name in SPLITS},
        set(read_split(LABELS_PATH)["code"]),
    )
    out = args.processed_dir / FEEDBACK_TRAIN_PATH.name
    write_split(accepted, out)
    write_split(removed, out.with_name("feedback_removed.csv"))
    report["source_file"] = args.feedback_csv.name
    write_json(report, out.with_name("feedback_report.json"))

    print(f"  {report['input']} báo cáo -> nhận {report['accepted']}, loại {len(removed)}")
    for reason, n in report["removed"].items():
        if n:
            print(f"    {reason:<18}{n:>6}")
    print(f"Đã ghi {rel(out)}")


if __name__ == "__main__":
    main()
