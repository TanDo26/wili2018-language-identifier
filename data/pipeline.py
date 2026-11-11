"""Chạy toàn bộ pipeline dữ liệu: tải -> làm sạch -> chia -> kiểm tra rò rỉ -> cắt ngắn -> thống kê.

    python -m data.pipeline               # dùng lại data/splits/split_ids.csv đã commit
    python -m data.pipeline --eda         # chạy thêm EDA cho dữ liệu thô và dữ liệu đã chia
    python -m data.pipeline --regenerate  # chia lại (seed 42) và ghi đè split_ids.csv
"""
from __future__ import annotations

import argparse

from data import augment, check_leakage, clean, download, eda, split, stats
from data.common import setup_console


def main(argv=None):
    setup_console()
    parser = argparse.ArgumentParser(description="Pipeline dữ liệu WiLI-2018")
    parser.add_argument("--eda", action="store_true", help="chạy thêm EDA")
    parser.add_argument("--regenerate", action="store_true", help="chia lại split thay vì dùng split_ids.csv")
    args = parser.parse_args(argv)

    steps = [("download", download.main, [])]
    if args.eda:
        steps.append(("eda (raw)", eda.main, ["--stage", "raw"]))
    steps += [
        ("clean", clean.main, []),
        ("split", split.main, ["--regenerate"] if args.regenerate else []),
        ("check_leakage", check_leakage.main, []),
        ("augment", augment.main, []),
        ("stats", stats.main, []),
    ]
    if args.eda:
        steps.append(("eda (processed)", eda.main, ["--stage", "processed"]))

    for i, (name, fn, step_args) in enumerate(steps, start=1):
        print(f"\n[{i}/{len(steps)}] {name}")
        fn(step_args)


if __name__ == "__main__":
    main()
