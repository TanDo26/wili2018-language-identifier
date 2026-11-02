"""Tổng hợp thống kê bộ dữ liệu -> data/processed/stats.json.

    python -m data.stats

stats.json là thỏa thuận chung của nhóm về dữ liệu: B dùng để kiểm tra đúng dữ liệu
trước khi train, C/D dùng để hiển thị (danh sách ngôn ngữ, số mẫu…). Cấu trúc file
xem data/stats.example.json.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data.common import (
    CLEAN_REPORT_PATH,
    LABELS_PATH,
    LEAKAGE_REPORT_PATH,
    PROCESSED_DIR,
    SPLIT_IDS_PATH,
    SPLITS,
    STATS_PATH,
    file_hash,
    length_stats,
    load_split,
    read_json,
    read_split,
    rel,
    setup_console,
    write_json,
)
from data.download import DOI, EXPECTED_MD5, LICENSE, RECORD_URL

SCHEMA_VERSION = "1.0"


def build_stats(frames: dict[str, pd.DataFrame], labels: pd.DataFrame, clean_report: dict,
                leakage_report: dict, split_meta: dict, processed_dir: Path = PROCESSED_DIR) -> dict:
    counts = {name: df["label"].value_counts() for name, df in frames.items()}
    info = labels.set_index("code")
    codes = sorted(set().union(*(c.index for c in counts.values())))

    splits = {}
    for name, df in frames.items():
        path = processed_dir / f"{name}.csv"
        splits[name] = {
            "file": rel(path),
            "sha256": file_hash(path),
            "num_samples": len(df),
            "min_per_language": int(counts[name].min()),
            "max_per_language": int(counts[name].max()),
            "length_chars": length_stats(df["text"].str.len()),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "name": "WiLI-2018",
            "version": "1.0.0",
            "source_url": RECORD_URL,
            "doi": DOI,
            "license": LICENSE,
            "raw_md5": EXPECTED_MD5,
        },
        "split": {
            "seed": split_meta["seed"],
            "val_per_language": split_meta["val_per_lang"],
            "val_max_fraction": split_meta["val_max_fraction"],
            "val_short_languages": split_meta.get("val_short_languages", {}),
            "split_ids_file": rel(SPLIT_IDS_PATH),
            "split_ids_sha256": split_meta["split_ids_sha256"],
        },
        "num_languages": len(codes),
        "splits": splits,
        "cleaning": clean_report,
        "leakage": {
            "passed": leakage_report["passed"],
            "pairs": {pair: {check: v["count"] for check, v in checks.items()}
                      for pair, checks in leakage_report["pairs"].items()},
        },
        "languages": [
            {
                "code": code,
                "name": info["name"].get(code, ""),
                "family": info["family"].get(code, ""),
                **{name: int(counts[name].get(code, 0)) for name in frames},
            }
            for code in codes
        ],
    }


def main(argv=None):
    setup_console()
    parser = argparse.ArgumentParser(description="Tạo data/processed/stats.json")
    parser.add_argument("--out", type=Path, default=STATS_PATH)
    args = parser.parse_args(argv)

    missing = [p for p in (CLEAN_REPORT_PATH, LEAKAGE_REPORT_PATH, LABELS_PATH,
                           SPLIT_IDS_PATH.with_name("split_meta.json")) if not p.exists()]
    if missing:
        raise SystemExit(f"Thiếu {[rel(p) for p in missing]} – chạy clean, split, check_leakage trước.")

    stats = build_stats(
        frames={name: load_split(name) for name in SPLITS},
        labels=read_split(LABELS_PATH),
        clean_report=read_json(CLEAN_REPORT_PATH),
        leakage_report=read_json(LEAKAGE_REPORT_PATH),
        split_meta=read_json(SPLIT_IDS_PATH.with_name("split_meta.json")),
    )
    write_json(stats, args.out)
    sizes = ", ".join(f"{k}={v['num_samples']:,}" for k, v in stats["splits"].items())
    print(f"{stats['num_languages']} ngôn ngữ; {sizes}; leakage passed={stats['leakage']['passed']}")
    print(f"Đã ghi {rel(args.out)}")


if __name__ == "__main__":
    main()
