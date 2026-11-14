"""Làm sạch dữ liệu thô data/raw/wili-2018/ -> data/interim/.

    python -m data.clean

Các bước (theo thứ tự):
  1. Chuẩn hóa văn bản: Unicode NFC, bỏ ký tự điều khiển / BOM, gộp khoảng trắng.
  2. Bỏ dòng rỗng và dòng không chứa chữ cái nào (chỉ số / ký hiệu).
  3. Bỏ toàn bộ các đoạn bị gán nhiều nhãn khác nhau (xét chung train + test).
  4. Bỏ trùng lặp trong từng tập, giữ lần xuất hiện đầu tiên.
  5. Bỏ khỏi train các đoạn đã có trong test (giữ nguyên test).

"Trùng" được xác định bằng near_dup_key (casefold, chỉ giữ chữ cái + dấu), nên các
đoạn chỉ khác nhau ở chữ số / dấu câu (bài viết do bot sinh theo mẫu) cũng bị gộp.
Bước 4 áp dụng cả cho train vì tập validation được tách ra từ train.

Đầu ra:
  data/interim/train.csv, test.csv   (id, text, label)
  data/interim/removed.csv           (id, label, reason, text) – các dòng bị loại
  data/interim/clean_report.json     số dòng bị loại ở từng bước
  data/processed/labels.csv          bảng mã ngôn ngữ -> tên, họ ngôn ngữ
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data.common import (
    CLEAN_REPORT_PATH,
    INTERIM_DIR,
    LABELS_PATH,
    WILI_DIR,
    make_ids,
    near_dup_key,
    normalize_text,
    read_lines,
    rel,
    setup_console,
    write_json,
    write_split,
)

SOURCES = ("train", "test")
REASONS = ("empty", "no_letters", "conflicting_labels", "duplicate_within_split", "overlap_with_test")


def load_raw(wili_dir: Path = WILI_DIR) -> dict[str, pd.DataFrame]:
    frames = {}
    for source in SOURCES:
        x = read_lines(wili_dir / f"x_{source}.txt")
        y = read_lines(wili_dir / f"y_{source}.txt")
        if len(x) != len(y):
            raise ValueError(f"x_{source}.txt ({len(x)} dòng) và y_{source}.txt ({len(y)} dòng) lệch nhau")
        frames[source] = pd.DataFrame({"id": make_ids(source, len(x)), "text": x, "label": y})
    return frames


def load_labels(wili_dir: Path = WILI_DIR) -> pd.DataFrame:
    df = pd.read_csv(wili_dir / "labels.csv", sep=";", dtype=str,
                     keep_default_na=False, na_filter=False, encoding="utf-8")
    df = df.rename(columns={
        "Label": "code",
        "English": "name",
        "Wiki Code": "wiki_code",
        "ISO 369-3": "iso639_3",  # tên cột gốc bị gõ nhầm "369"
        "Language family": "family",
        "Writing system": "writing_system",
    })
    df = df[["code", "name", "wiki_code", "iso639_3", "family", "writing_system"]]
    return df.apply(lambda col: col.str.strip())


def clean(frames: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict]:
    """Làm sạch train/test. Trả về (frames đã sạch, các dòng bị loại, report)."""
    removed = []
    counts = {reason: {source: 0 for source in SOURCES} for reason in REASONS}

    def drop(source: str, df: pd.DataFrame, mask: pd.Series, reason: str) -> pd.DataFrame:
        if mask.any():
            removed.append(df.loc[mask, ["id", "label", "text"]].assign(reason=reason))
        counts[reason][source] = int(mask.sum())
        return df.loc[~mask]

    work = {}
    for source in SOURCES:
        df = frames[source].copy()
        df["text"] = df["text"].map(normalize_text)
        df["key"] = df["text"].map(near_dup_key)
        df = drop(source, df, df["text"] == "", "empty")
        df = drop(source, df, df["key"] == "", "no_letters")
        work[source] = df

    both = pd.concat(work.values())
    n_labels = both.groupby("key")["label"].nunique()
    conflicting = set(n_labels.index[n_labels > 1])
    for source in SOURCES:
        df = work[source]
        df = drop(source, df, df["key"].isin(conflicting), "conflicting_labels")
        df = df.sort_values("id")
        df = drop(source, df, df["key"].duplicated(keep="first"), "duplicate_within_split")
        work[source] = df

    test_keys = set(work["test"]["key"])
    work["train"] = drop("train", work["train"], work["train"]["key"].isin(test_keys), "overlap_with_test")

    cleaned = {s: df[["id", "text", "label"]].reset_index(drop=True) for s, df in work.items()}
    removed_df = (pd.concat(removed) if removed else pd.DataFrame(columns=["id", "label", "text", "reason"]))
    removed_df = removed_df[["id", "label", "reason", "text"]].sort_values("id").reset_index(drop=True)

    report = {
        "raw": {s: len(frames[s]) for s in SOURCES},
        "removed": counts,
        "removed_total": {s: sum(counts[r][s] for r in REASONS) for s in SOURCES},
        "clean": {s: len(cleaned[s]) for s in SOURCES},
        "conflicting_texts": len(conflicting),
    }
    return cleaned, removed_df, report


def main(argv=None):
    setup_console()
    parser = argparse.ArgumentParser(description="Làm sạch WiLI-2018")
    parser.add_argument("--raw-dir", type=Path, default=WILI_DIR)
    parser.add_argument("--out-dir", type=Path, default=INTERIM_DIR)
    args = parser.parse_args(argv)

    if not (args.raw_dir / "x_train.txt").exists():
        raise SystemExit(f"Không thấy dữ liệu thô ở {args.raw_dir} – chạy `python -m data.download` trước.")

    frames = load_raw(args.raw_dir)
    cleaned, removed, report = clean(frames)

    for source, df in cleaned.items():
        write_split(df, args.out_dir / f"{source}.csv")
    write_split(removed, args.out_dir / "removed.csv")
    report_path = args.out_dir / CLEAN_REPORT_PATH.name
    write_json(report, report_path)

    labels = load_labels(args.raw_dir)
    used = set(cleaned["train"]["label"]) | set(cleaned["test"]["label"])
    write_split(labels[labels["code"].isin(used)].sort_values("code"), LABELS_PATH)

    print(f"{'bước':<24}{'train':>8}{'test':>8}")
    for reason in REASONS:
        print(f"  {reason:<22}{report['removed'][reason]['train']:>8}{report['removed'][reason]['test']:>8}")
    print(f"{'raw':<24}{report['raw']['train']:>8}{report['raw']['test']:>8}")
    print(f"{'clean':<24}{report['clean']['train']:>8}{report['clean']['test']:>8}")
    print(f"Đã ghi {rel(args.out_dir)}/ và {rel(LABELS_PATH)}")


if __name__ == "__main__":
    main()
