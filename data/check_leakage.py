"""Kiểm tra rò rỉ dữ liệu giữa các tập train / val / test.

    python -m data.check_leakage       # exit code 1 nếu phát hiện rò rỉ

Với mỗi cặp tập (train-val, train-test, val-test) kiểm tra:
  - id: cùng một mẫu nằm ở hai tập;
  - exact: văn bản giống hệt nhau;
  - near: giống nhau sau khi bỏ chữ số, dấu câu, khoảng trắng, hoa/thường (near_dup_key).
Ngoài ra kiểm tra tập nhãn của các tập giống nhau, nguồn gốc id đúng (test lấy từ
x_test.txt) và các file đã chia khớp với data/splits/split_ids.csv.

Kết quả ghi vào data/processed/leakage_report.json.
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd

from data.common import (
    ID_PREFIX,
    LEAKAGE_REPORT_PATH,
    PROCESSED_DIR,
    SPLIT_IDS_PATH,
    SPLITS,
    load_split,
    near_dup_key,
    rel,
    setup_console,
    write_json,
)

MAX_EXAMPLES = 5


def _overlap(a: pd.Series, b: pd.Series, a_ids: pd.Series) -> tuple[int, list[str]]:
    """Số mẫu của tập a có giá trị xuất hiện trong tập b (kèm vài id làm ví dụ)."""
    mask = a.isin(set(b))
    return int(mask.sum()), a_ids[mask].head(MAX_EXAMPLES).tolist()


def find_leaks(frames: dict[str, pd.DataFrame], split_ids: pd.DataFrame | None = None) -> dict:
    keys = {name: df["text"].map(near_dup_key) for name, df in frames.items()}
    report = {"pairs": {}, "within_split_duplicates": {}, "labels": {}, "sources": {}, "errors": []}

    for a, b in combinations(SPLITS, 2):
        fa, fb = frames[a], frames[b]
        pair = {}
        for check, col_a, col_b in (
            ("id", fa["id"], fb["id"]),
            ("exact", fa["text"], fb["text"]),
            ("near", keys[a], keys[b]),
        ):
            n, examples = _overlap(col_a, col_b, fa["id"])
            pair[check] = {"count": n, "examples": examples}
            if n:
                report["errors"].append(f"{a}-{b}: {n} mẫu trùng ({check})")
        report["pairs"][f"{a}-{b}"] = pair

    # Trùng lặp trong cùng một tập không phải rò rỉ, nhưng là dấu hiệu bước làm sạch bị bỏ qua.
    for name, df in frames.items():
        report["within_split_duplicates"][name] = {
            "exact": int(df["text"].duplicated().sum()),
            "near": int(keys[name].duplicated().sum()),
        }

    label_sets = {name: set(df["label"]) for name, df in frames.items()}
    all_labels = set().union(*label_sets.values())
    for name, labels in label_sets.items():
        missing = sorted(all_labels - labels)
        report["labels"][name] = {"num_languages": len(labels), "missing": missing}
        if missing:
            report["errors"].append(f"{name} thiếu {len(missing)} ngôn ngữ: {missing[:10]}")

    for name, df in frames.items():
        expected = ID_PREFIX["test" if name == "test" else "train"] + "_"
        wrong = int((~df["id"].str.startswith(expected)).sum())
        report["sources"][name] = {"expected_prefix": expected, "wrong": wrong}
        if wrong:
            report["errors"].append(f"{name}: {wrong} mẫu không đến từ file gốc mong đợi ({expected}*)")

    if split_ids is not None:
        saved = split_ids.set_index("id")["split"]
        actual = pd.concat([pd.Series(name, index=df["id"]) for name, df in frames.items()])
        mismatched = (int((actual.reindex(saved.index) != saved).sum())
                      + len(actual.index.difference(saved.index)))
        report["split_ids"] = {"mismatched": mismatched}
        if mismatched:
            report["errors"].append(f"{mismatched} mẫu không khớp split_ids.csv")

    report["passed"] = not report["errors"]
    return report


def main(argv=None):
    setup_console()
    parser = argparse.ArgumentParser(description="Kiểm tra rò rỉ dữ liệu")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--split-ids", type=Path, default=SPLIT_IDS_PATH)
    args = parser.parse_args(argv)

    frames = {name: load_split(name, args.processed_dir) for name in SPLITS}
    split_ids = None
    if args.split_ids.exists():
        split_ids = pd.read_csv(args.split_ids, dtype=str, keep_default_na=False)
    report = find_leaks(frames, split_ids)
    out = args.processed_dir / LEAKAGE_REPORT_PATH.name
    write_json(report, out)

    for pair, checks in report["pairs"].items():
        summary = ", ".join(f"{c}={v['count']}" for c, v in checks.items())
        print(f"  {pair:<11} {summary}")
    if report["passed"]:
        print(f"Không phát hiện rò rỉ. Báo cáo: {rel(out)}")
    else:
        for err in report["errors"]:
            print(f"  LỖI: {err}")
        raise SystemExit(f"Phát hiện rò rỉ – xem {rel(out)}")


if __name__ == "__main__":
    main()
