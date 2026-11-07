"""EDA cho WiLI-2018: độ dài văn bản, phân bố nhãn, trùng lặp, hệ chữ.

    python -m data.eda                     # dữ liệu thô (sau python -m data.download)
    python -m data.eda --stage processed   # dữ liệu đã làm sạch + chia (sau python -m data.split)

Đầu ra trong data/reports/: eda_<stage>.json, eda_<stage>.md và hình figures/<stage>_*.png
(hình chỉ được vẽ khi đã cài matplotlib).
"""
from __future__ import annotations

import argparse
import functools
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from data.common import (
    LABELS_PATH,
    REPORTS_DIR,
    SEED,
    SPLITS,
    WILI_DIR,
    length_stats,
    load_split,
    read_split,
    rel,
    setup_console,
    write_json,
)

LENGTH_BUCKETS = [(0, 200), (200, 500), (500, 1000), (1000, 5000), (5000, None)]
HIST_RANGE = (100, 10_000)  # văn bản dài hơn được gộp vào cột cuối của histogram
SCRIPT_SAMPLE_PER_LANG = 30

# Bảng màu (đã kiểm tra mù màu): train / val / test luôn giữ cùng một màu.
COLORS = {"train": "#2a78d6", "test": "#eb6834", "val": "#1baf7a"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


@functools.lru_cache(maxsize=None)
def script_of(ch: str) -> str | None:
    """Hệ chữ của một chữ cái, lấy từ tên Unicode (vd. "LATIN SMALL LETTER A" -> "Latin")."""
    if not unicodedata.category(ch).startswith("L"):
        return None
    name = unicodedata.name(ch, "")
    if not name:
        return None
    script = name.split(" ")[0]
    return script if script == "CJK" else script.title()


def dominant_script(texts) -> str:
    counter = Counter()
    for text in texts:
        counter.update(s for s in map(script_of, text) if s)
    return counter.most_common(1)[0][0] if counter else "Unknown"


def bucket_label(lo: int, hi: int | None) -> str:
    return f">={lo}" if hi is None else f"{lo}-{hi - 1}"


def load_frames(stage: str) -> dict[str, pd.DataFrame]:
    if stage == "raw":
        from data.clean import load_raw
        return load_raw(WILI_DIR)
    return {name: load_split(name) for name in SPLITS}


def load_label_info(stage: str) -> pd.DataFrame:
    if stage == "processed" and LABELS_PATH.exists():
        return read_split(LABELS_PATH)
    from data.clean import load_labels
    return load_labels(WILI_DIR)


def analyze(frames: dict[str, pd.DataFrame], labels: pd.DataFrame, seed: int = SEED) -> dict:
    report = {"splits": {}}
    lengths = {name: df["text"].str.len() for name, df in frames.items()}

    for name, df in frames.items():
        counts = df["label"].value_counts()
        n = len(df)
        buckets = {}
        for lo, hi in LENGTH_BUCKETS:
            mask = (lengths[name] >= lo) & ((lengths[name] < hi) if hi else True)
            buckets[bucket_label(lo, hi)] = round(float(mask.sum()) / n, 4) if n else 0.0
        report["splits"][name] = {
            "num_samples": n,
            "num_languages": int(counts.size),
            "per_language": {
                "min": int(counts.min()),
                "max": int(counts.max()),
                "mean": round(float(counts.mean()), 2),
                "fewest": {k: int(v) for k, v in counts.sort_values().head(10).items()},
            },
            "length_chars": length_stats(lengths[name]),
            "length_buckets": buckets,
            "exact_duplicates": int(df["text"].duplicated().sum()),
            "not_nfc": int(sum(not unicodedata.is_normalized("NFC", t) for t in df["text"])),
        }

    all_df = pd.concat([df.assign(split=name) for name, df in frames.items()], ignore_index=True)
    all_df["length"] = all_df["text"].str.len()

    if "train" in frames and "test" in frames:
        test_texts = set(frames["test"]["text"])
        n_labels = all_df.groupby("text")["label"].nunique()
        report["overlap"] = {
            "train_rows_in_test": int(frames["train"]["text"].isin(test_texts).sum()),
            "conflicting_label_texts": int((n_labels > 1).sum()),
        }

    rng = np.random.default_rng(seed)
    per_lang = []
    info = labels.set_index("code")
    for label, group in all_df.groupby("label"):
        texts = group["text"].to_numpy()
        sample = rng.choice(texts, size=min(SCRIPT_SAMPLE_PER_LANG, len(texts)), replace=False)
        row = {
            "code": label,
            "name": info["name"].get(label, ""),
            "family": info["family"].get(label, ""),
            "script": dominant_script(sample),
            "median_length": float(group["length"].median()),
        }
        row.update({name: int((group["split"] == name).sum()) for name in frames})
        per_lang.append(row)
    lang_df = pd.DataFrame(per_lang).sort_values("code")
    report["languages"] = lang_df.to_dict(orient="records")
    report["scripts"] = {k: int(v) for k, v in lang_df["script"].value_counts().items()}
    families = lang_df["family"].replace("", "Unknown").value_counts()
    report["families"] = {k: int(v) for k, v in families.items()}
    return report


def to_markdown(report: dict, stage: str, figures: list[Path]) -> str:
    splits = report["splits"]
    lines = [f"# EDA WiLI-2018 ({stage})", ""]

    lines += ["## Tổng quan", "",
              "| Tập | Số mẫu | Số ngôn ngữ | Mẫu/ngôn ngữ (min–max) | Trùng lặp exact | Không ở dạng NFC |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, s in splits.items():
        pl = s["per_language"]
        lines.append(f"| {name} | {s['num_samples']:,} | {s['num_languages']} | {pl['min']}–{pl['max']} "
                     f"| {s['exact_duplicates']:,} | {s['not_nfc']} |")
    if "overlap" in report:
        o = report["overlap"]
        lines += ["", f"- Số dòng train trùng y hệt một dòng test: **{o['train_rows_in_test']:,}**",
                  f"- Số đoạn văn bị gán nhiều nhãn khác nhau: **{o['conflicting_label_texts']:,}**"]

    lines += ["", "## Độ dài văn bản (số ký tự)", "",
              "| Tập | min | p5 | median | mean | p95 | max |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, s in splits.items():
        L = s["length_chars"]
        lines.append(f"| {name} | {L['min']} | {L['p5']:.0f} | {L['median']:.0f} | {L['mean']:.0f} "
                     f"| {L['p95']:.0f} | {L['max']:,} |")
    bucket_names = list(next(iter(splits.values()))["length_buckets"])
    lines += ["", "| Tập | " + " | ".join(bucket_names) + " |", "|---|" + "---:|" * len(bucket_names)]
    for name, s in splits.items():
        lines.append(f"| {name} | " + " | ".join(f"{v:.1%}" for v in s["length_buckets"].values()) + " |")

    lang_df = pd.DataFrame(report["languages"])
    first = next(iter(splits))
    lines += ["", f"## Ngôn ngữ ít mẫu nhất (tập {first})", "", "| Mã | Tên | Số mẫu |", "|---|---|---:|"]
    for _, r in lang_df.nsmallest(10, first).iterrows():
        lines.append(f"| {r['code']} | {r['name']} | {r[first]} |")

    lines += ["", "## Độ dài trung vị theo ngôn ngữ", "", "| Ngắn nhất | median | Dài nhất | median |",
              "|---|---:|---|---:|"]
    shortest = lang_df.nsmallest(10, "median_length").itertuples()
    longest = lang_df.nlargest(10, "median_length").itertuples()
    for a, b in zip(shortest, longest):
        lines.append(f"| {a.name} ({a.code}) | {a.median_length:.0f} | {b.name} ({b.code}) | {b.median_length:.0f} |")

    lines += ["", "## Hệ chữ chiếm ưu thế (số ngôn ngữ)", "", "| Hệ chữ | Số ngôn ngữ |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in report["scripts"].items()]

    lines += ["", "## Họ ngôn ngữ (10 họ lớn nhất)", "", "| Họ | Số ngôn ngữ |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in list(report["families"].items())[:10]]

    if figures:
        lines += ["", "## Hình", ""]
        lines += [f"![{p.stem}](figures/{p.name})" for p in figures]
    return "\n".join(lines) + "\n"


def _style_axes(ax, grid_axis: str = "y") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_figures(frames: dict[str, pd.DataFrame], report: dict, stage: str, out_dir: Path) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Chưa cài matplotlib – bỏ qua phần vẽ hình.")
        return []

    plt.rcParams.update({"figure.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK_2,
                         "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left"})
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    lang_df = pd.DataFrame(report["languages"])

    def save(fig, name):
        path = out_dir / f"{stage}_{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    # 1. Phân bố độ dài (tỉ lệ, trục log) – đường bậc thang để các tập chồng lên nhau vẫn đọc được.
    fig, ax = plt.subplots(figsize=(8, 4))
    lo, hi = HIST_RANGE
    bins = np.logspace(np.log10(lo), np.log10(hi), 50)
    for name, df in frames.items():
        clipped = df["text"].str.len().clip(lo, hi - 1)
        ax.hist(clipped, bins=bins, weights=np.full(len(df), 1 / len(df)),
                histtype="step", linewidth=2, color=COLORS[name], label=f"{name} (n={len(df):,})")
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_xlabel(f"Độ dài văn bản (số ký tự, thang log; >{hi:,} gộp vào cột cuối)")
    ax.set_ylabel("Tỉ lệ mẫu")
    ax.set_title("Phân bố độ dài văn bản")
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2)
    save(fig, "length_hist")

    # 2. Số mẫu mỗi ngôn ngữ: toàn bộ (sắp xếp giảm dần) + 10 ngôn ngữ ít mẫu nhất.
    first = next(iter(frames))
    counts = lang_df.set_index("code")[first].sort_values(ascending=False)
    fig, (ax, ax_low) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [2.4, 1]})
    ax.bar(np.arange(len(counts)), counts.to_numpy(), width=0.8, color=COLORS[first], linewidth=0)
    ax.set_xticks([])
    ax.set_xlim(-1, len(counts))
    ax.set_xlabel(f"{len(counts)} ngôn ngữ (sắp xếp theo số mẫu)")
    ax.set_ylabel("Số mẫu")
    ax.set_title(f"Số mẫu mỗi ngôn ngữ – tập {first}")
    _style_axes(ax)

    low = counts.tail(10)
    names = lang_df.set_index("code")["name"]
    ax_low.barh([f"{names[c]} ({c})" for c in low.index], low.to_numpy(),
                height=0.7, color=COLORS[first], linewidth=0)
    for y, v in enumerate(low.to_numpy()):
        ax_low.text(v, y, f" {v}", va="center", fontsize=8, color=INK_2)
    ax_low.set_title("10 ngôn ngữ ít mẫu nhất")
    ax_low.set_xlabel("Số mẫu")
    _style_axes(ax_low, grid_axis="x")
    fig.tight_layout()
    save(fig, "label_counts")

    # 3. Ngôn ngữ có văn bản ngắn nhất / dài nhất (độ dài trung vị).
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, (title, sub) in zip(axes, (
        ("15 ngôn ngữ có văn bản ngắn nhất", lang_df.nsmallest(15, "median_length")),
        ("15 ngôn ngữ có văn bản dài nhất", lang_df.nlargest(15, "median_length")),
    )):
        sub = sub.iloc[::-1]
        ax.barh([f"{n} ({c})" for n, c in zip(sub["name"], sub["code"])], sub["median_length"],
                height=0.7, color=COLORS["train"], linewidth=0)
        ax.set_title(title)
        ax.set_xlabel("Độ dài trung vị (ký tự)")
        _style_axes(ax, grid_axis="x")
    fig.tight_layout()
    save(fig, "median_length")

    # 4. Số ngôn ngữ theo hệ chữ chiếm ưu thế.
    scripts = pd.Series(report["scripts"]).sort_values()
    fig, ax = plt.subplots(figsize=(7, max(3, 0.28 * len(scripts))))
    ax.barh(scripts.index, scripts.to_numpy(), height=0.7, color=COLORS["train"], linewidth=0)
    for y, v in enumerate(scripts.to_numpy()):
        ax.text(v, y, f" {v}", va="center", fontsize=8, color=INK_2)
    ax.set_xlabel("Số ngôn ngữ")
    ax.set_title("Hệ chữ chiếm ưu thế của từng ngôn ngữ")
    _style_axes(ax, grid_axis="x")
    save(fig, "scripts")

    return paths


def main(argv=None):
    setup_console()
    parser = argparse.ArgumentParser(description="EDA WiLI-2018")
    parser.add_argument("--stage", choices=("raw", "processed"), default="raw")
    parser.add_argument("--out-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    frames = load_frames(args.stage)
    report = analyze(frames, load_label_info(args.stage))
    report["stage"] = args.stage
    figures = [] if args.no_plots else plot_figures(frames, report, args.stage, args.out_dir / "figures")

    write_json(report, args.out_dir / f"eda_{args.stage}.json")
    md_path = args.out_dir / f"eda_{args.stage}.md"
    md_path.write_text(to_markdown(report, args.stage, figures), encoding="utf-8", newline="\n")
    print(f"Đã ghi {rel(md_path)}, {rel(args.out_dir / f'eda_{args.stage}.json')} và {len(figures)} hình.")


if __name__ == "__main__":
    main()
