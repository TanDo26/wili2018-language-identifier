"""Kiểm tra rò rỉ dữ liệu giữa các tập train / val / test.

    python -m data.check_leakage       # exit code 1 nếu phát hiện rò rỉ

Với mỗi cặp tập (train-val, train-test, val-test) kiểm tra:
  - id: cùng một mẫu nằm ở hai tập;
  - exact: văn bản giống hệt nhau;
  - near: giống nhau sau khi bỏ chữ số, dấu câu, khoảng trắng, hoa/thường (near_dup_key).
Ngoài ra kiểm tra tập nhãn của các tập giống nhau, nguồn gốc id đúng (test lấy từ
x_test.txt) và các file đã chia khớp với data/splits/split_ids.csv.

Các kiểm tra trên chỉ so sánh nguyên đoạn. Vì vậy có thêm phép đo phần văn bản val/test chung
với train: WiLI có nhiều bài do bot tạo theo mẫu, chỉ khác nhau ở tên riêng, nên không bị coi là
trùng. Phép đo này chỉ để báo cáo, không làm kiểm tra thất bại: bỏ các đoạn đó khỏi train sẽ mất
13,4% dữ liệu và nhiều ngôn ngữ gần như hết mẫu, nên nhóm giữ nguyên cách chia và báo cáo thêm
điểm trên phần test không có văn bản chung (experiments/error_analysis.py).

Kết quả ghi vào data/processed/leakage_report.json; tỉ lệ của từng đoạn có phần chung ghi vào
data/processed/overlap_with_train.csv (id, split, shared_fraction).
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd

from data.common import (
    ID_PREFIX,
    LEAKAGE_REPORT_PATH,
    OVERLAP_PATH,
    PROCESSED_DIR,
    SPLIT_IDS_PATH,
    SPLITS,
    load_split,
    near_dup_key,
    rel,
    setup_console,
    write_json,
    write_split,
)

MAX_EXAMPLES = 5

# Train được lấy cửa sổ OVERLAP_WINDOW ký tự, cách nhau OVERLAP_STRIDE ký tự; đoạn cần đo được
# dò ở mọi vị trí. Nhờ vậy mọi phần chung dài từ OVERLAP_WINDOW + OVERLAP_STRIDE - 1 ký tự đều
# chắc chắn được phát hiện.
OVERLAP_WINDOW = 40
OVERLAP_STRIDE = 20
OVERLAP_THRESHOLDS = (25, 50, 80)


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


def window_hashes(texts) -> set[int]:
    windows = set()
    for text in texts:
        windows.update(hash(text[i:i + OVERLAP_WINDOW])
                       for i in range(0, len(text) - OVERLAP_WINDOW + 1, OVERLAP_STRIDE))
    return windows


def shared_fraction(text: str, windows: set[int]) -> float:
    """Tỉ lệ ký tự của text nằm trong ít nhất một cửa sổ có ở `windows`.

    Đây là cận dưới của phần chung thật: mỗi đầu phần chung có thể thiếu tới OVERLAP_STRIDE - 1
    ký tự, vì train không được lấy cửa sổ ở mọi vị trí.
    """
    covered = end = 0
    for i in range(len(text) - OVERLAP_WINDOW + 1):
        if hash(text[i:i + OVERLAP_WINDOW]) in windows:
            covered += i + OVERLAP_WINDOW - max(i, end)
            end = i + OVERLAP_WINDOW
    return covered / max(len(text), 1)


def partial_overlap(frames: dict[str, pd.DataFrame]) -> tuple[dict, pd.DataFrame]:
    """Đo phần văn bản val/test chung với train: trả về (tóm tắt, tỉ lệ của từng đoạn có phần chung)."""
    windows = window_hashes(frames["train"]["text"])
    summary = {"window_chars": OVERLAP_WINDOW,
               "always_detected_from_chars": OVERLAP_WINDOW + OVERLAP_STRIDE - 1}
    rows = []
    for name in ("val", "test"):
        df = frames[name]
        shared = df["text"].map(lambda text: shared_fraction(text, windows)).astype(float)
        summary[name] = {
            "num_samples": len(df),
            "any": int((shared > 0).sum()),
            **{f"at_least_{pct}pct": int((shared >= pct / 100).sum()) for pct in OVERLAP_THRESHOLDS},
        }
        has_shared = shared > 0
        rows.append(pd.DataFrame({"id": df.loc[has_shared, "id"], "split": name,
                                  "shared_fraction": shared[has_shared].round(4)}))
    return summary, pd.concat(rows, ignore_index=True)


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
    report["partial_overlap"], overlap_rows = partial_overlap(frames)
    out = args.processed_dir / LEAKAGE_REPORT_PATH.name
    write_json(report, out)
    write_split(overlap_rows, args.processed_dir / OVERLAP_PATH.name)

    for pair, checks in report["pairs"].items():
        summary = ", ".join(f"{c}={v['count']}" for c, v in checks.items())
        print(f"  {pair:<11} {summary}")
    for name in ("val", "test"):
        part = report["partial_overlap"][name]
        print(f"  {name:<11} có phần chung với train: {part['any']:,}/{part['num_samples']:,} đoạn "
              f"({part['any'] / part['num_samples']:.2%}), từ 50% độ dài trở lên: {part['at_least_50pct']:,} "
              "(chỉ để báo cáo)")
    if report["passed"]:
        print(f"Không phát hiện rò rỉ. Báo cáo: {rel(out)}")
    else:
        for err in report["errors"]:
            print(f"  LỖI: {err}")
        raise SystemExit(f"Phát hiện rò rỉ – xem {rel(out)}")


if __name__ == "__main__":
    main()
