"""Chia dữ liệu đã làm sạch thành train / val / test -> data/processed/.

    python -m data.split               # dùng lại data/splits/split_ids.csv nếu đã có
    python -m data.split --regenerate  # chia lại từ đầu (seed 42) và ghi đè split_ids

- test: toàn bộ x_test.txt của WiLI sau khi làm sạch (giữ nguyên test chuẩn).
- val : lấy ngẫu nhiên từ train gốc, mỗi ngôn ngữ VAL_PER_LANG (50) đoạn, tối đa
        VAL_MAX_FRACTION số đoạn còn lại của ngôn ngữ đó.
- train: phần còn lại của train gốc.

split_ids.csv (id, split) được commit lên git để mọi người tái lập đúng cùng một cách chia;
split_meta.json ghi seed và sha256 của các file đầu ra để đối chiếu.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data.common import (
    ID_PREFIX,
    INTERIM_DIR,
    PROCESSED_DIR,
    SEED,
    SPLIT_IDS_PATH,
    SPLITS,
    VAL_MAX_FRACTION,
    VAL_PER_LANG,
    file_hash,
    read_split,
    rel,
    setup_console,
    write_json,
    write_split,
)


def make_val_split(train: pd.DataFrame, val_per_lang: int = VAL_PER_LANG, seed: int = SEED,
                   max_fraction: float = VAL_MAX_FRACTION) -> pd.Series:
    """Trả về Series id -> "train"/"val", chọn val phân tầng theo nhãn.

    Nhãn và id được sắp xếp trước khi rút ngẫu nhiên nên kết quả chỉ phụ thuộc vào
    seed và tập id đầu vào, không phụ thuộc thứ tự dòng.
    """
    rng = np.random.default_rng(seed)
    assignment = pd.Series("train", index=train["id"].to_numpy(), dtype=object)
    for label, group in sorted(train.groupby("label")["id"], key=lambda kv: kv[0]):
        ids = np.sort(group.to_numpy())
        n_val = min(val_per_lang, int(len(ids) * max_fraction))
        if n_val > 0:
            assignment[rng.choice(ids, size=n_val, replace=False)] = "val"
    return assignment.sort_index()


def assign_splits(pool: pd.DataFrame, assignment: pd.Series) -> dict[str, pd.DataFrame]:
    split = pool["id"].map(assignment)
    return {s: pool[split == s].sort_values("id").reset_index(drop=True) for s in SPLITS}


def load_assignment(path: Path, pool: pd.DataFrame) -> pd.Series:
    ids = pd.read_csv(path, dtype=str, keep_default_na=False)
    if set(ids["split"]) - set(SPLITS):
        raise SystemExit(f"{path} có giá trị split lạ: {set(ids['split']) - set(SPLITS)}")
    if ids["id"].duplicated().any():
        raise SystemExit(f"{path} có id bị lặp")
    saved, current = set(ids["id"]), set(pool["id"])
    if saved != current:
        raise SystemExit(
            f"{rel(path)} không khớp dữ liệu sau làm sạch "
            f"({len(saved - current)} id thừa, {len(current - saved)} id thiếu). "
            "Nếu cố ý đổi bước làm sạch, chạy lại với --regenerate."
        )
    return ids.set_index("id")["split"].sort_index()


def check_sources(assignment: pd.Series) -> None:
    """test phải đến từ x_test.txt; train/val phải đến từ x_train.txt."""
    from_test = assignment.index.str.startswith(ID_PREFIX["test"] + "_")
    if (assignment[from_test] != "test").any() or (assignment[~from_test] == "test").any():
        raise SystemExit("Phân chia sai nguồn: tập test phải trùng đúng với x_test.txt")


def main(argv=None):
    setup_console()
    parser = argparse.ArgumentParser(description="Chia train / val / test")
    parser.add_argument("--in-dir", type=Path, default=INTERIM_DIR)
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--split-ids", type=Path, default=SPLIT_IDS_PATH)
    parser.add_argument("--regenerate", action="store_true", help="chia lại và ghi đè split_ids")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--val-per-lang", type=int, default=VAL_PER_LANG)
    args = parser.parse_args(argv)

    train = read_split(args.in_dir / "train.csv")
    test = read_split(args.in_dir / "test.csv")
    pool = pd.concat([train, test], ignore_index=True)

    if args.split_ids.exists() and not args.regenerate:
        print(f"Dùng lại cách chia trong {rel(args.split_ids)}")
        assignment = load_assignment(args.split_ids, pool)
    else:
        print(f"Chia mới: seed={args.seed}, val_per_lang={args.val_per_lang}")
        assignment = pd.concat([
            make_val_split(train, args.val_per_lang, args.seed),
            pd.Series("test", index=test["id"].to_numpy(), dtype=object),
        ]).sort_index()
        args.split_ids.parent.mkdir(parents=True, exist_ok=True)
        assignment.rename_axis("id").rename("split").to_csv(
            args.split_ids, header=True, encoding="utf-8", lineterminator="\n")
    check_sources(assignment)

    frames = assign_splits(pool, assignment)
    meta = {
        "seed": args.seed,
        "val_per_lang": args.val_per_lang,
        "val_max_fraction": VAL_MAX_FRACTION,
        "split_ids_sha256": file_hash(args.split_ids),
        "splits": {},
    }
    for name, df in frames.items():
        path = args.out_dir / f"{name}.csv"
        write_split(df, path)
        counts = df["label"].value_counts()
        meta["splits"][name] = {
            "file": rel(path),
            "num_samples": len(df),
            "num_languages": int(counts.size),
            "sha256": file_hash(path),
        }
        print(f"  {name:<5} {len(df):>7} mẫu, {counts.size} ngôn ngữ, "
              f"mỗi ngôn ngữ {counts.min()}–{counts.max()}  -> {rel(path)}")

    short_val = frames["val"]["label"].value_counts()
    short_val = short_val[short_val < args.val_per_lang]
    if not short_val.empty:
        print(f"  Lưu ý: val có ít hơn {args.val_per_lang} mẫu ở {short_val.to_dict()}")
    meta["val_short_languages"] = {k: int(v) for k, v in short_val.items()}

    meta_path = args.split_ids.with_name("split_meta.json")
    write_json(meta, meta_path)
    print(f"Đã ghi {rel(meta_path)}")


if __name__ == "__main__":
    main()
