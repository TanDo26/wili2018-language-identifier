"""Phân tích lỗi theo độ dài văn bản và theo các cặp ngôn ngữ hay bị nhầm.

    python -m experiments.error_analysis experiments/output/predictions_<cấu hình>.csv [file khác ...]

File dự đoán (do `feature_search --final` tạo, hoặc evaluate.py của B xuất cùng định dạng):
    id, source_id, label, pred, bucket ("20" | "30" | "50" | "100" | "full"), n_chars, [confidence]

File đầu tiên là cấu hình chính: phân tích cặp nhầm, ngôn ngữ yếu và ngưỡng độ tin cậy.
Các file sau chỉ được vẽ thêm vào đường độ chính xác theo độ dài để so sánh.
Mọi file đều được tách điểm theo phần văn bản chung với train (data/check_leakage.py).
Kết quả: experiments/results/error_analysis.md, .json và figures/error_*.png.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from data.common import LABELS_PATH, OVERLAP_PATH, REPORTS_DIR, read_json, read_split, rel, setup_console, write_json

from experiments.feature_search import RESULTS_DIR

SHORT_BUCKETS = ["20", "30", "50", "100"]
NATURAL_BUCKETS = [(140, 200), (200, 500), (500, 1000), (1000, None)]
PAIR_BUCKETS = ["full", "20"]
THRESHOLDS = (0.4, 0.6, 0.8, 0.9)
TOP_PAIRS = 15

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def natural_label(lo: int, hi: int | None) -> str:
    return f"{lo}+" if hi is None else f"{lo}–{hi - 1}"


def length_groups(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Các nhóm độ dài: đoạn cắt ngắn 20/30/50/100 rồi đoạn đầy đủ chia theo độ dài thật."""
    groups = {f"{b} ký tự": df[df["bucket"] == b] for b in SHORT_BUCKETS}
    full = df[df["bucket"] == "full"]
    for lo, hi in NATURAL_BUCKETS:
        mask = (full["n_chars"] >= lo) & ((full["n_chars"] < hi) if hi else True)
        groups[natural_label(lo, hi)] = full[mask]
    groups["Toàn bộ test"] = full
    return {name: g for name, g in groups.items() if not g.empty}


def _scores(g: pd.DataFrame) -> dict:
    return {
        "n": len(g),
        "accuracy": round(float(accuracy_score(g["label"], g["pred"])), 4),
        "macro_f1": round(float(f1_score(g["label"], g["pred"], average="macro")), 4),
    }


def by_length(df: pd.DataFrame) -> dict:
    return {name: _scores(g) for name, g in length_groups(df).items()}


def by_overlap(df: pd.DataFrame, shared: pd.Series) -> dict:
    """Điểm trên các đoạn test đầy đủ, tách theo tỉ lệ văn bản chung với train (id -> tỉ lệ)."""
    full = df[df["bucket"] == "full"]
    fraction = full["id"].map(shared).fillna(0.0)
    groups = {"Toàn bộ test": full,
              "Không có văn bản chung với train": full[fraction == 0],
              "Chung từ 50% độ dài trở lên": full[fraction >= 0.5]}
    return {name: _scores(g) for name, g in groups.items() if not g.empty}


def load_shared_fraction() -> pd.Series | None:
    if not OVERLAP_PATH.exists():
        print(f"Chưa có {rel(OVERLAP_PATH)} – bỏ qua phần tách điểm theo văn bản chung.")
        return None
    rows = read_split(OVERLAP_PATH)
    rows = rows[rows["split"] == "test"]
    return rows.set_index("id")["shared_fraction"].astype(float)


def confused_pairs(df: pd.DataFrame, info: pd.DataFrame, top: int = TOP_PAIRS) -> list[dict]:
    """Cặp ngôn ngữ (không phân biệt chiều) có nhiều lỗi nhầm lẫn qua lại nhất."""
    support = df["label"].value_counts()
    err = df[df["label"] != df["pred"]]
    directed = err.groupby(["label", "pred"]).size()
    pairs = {}
    for (a, b), n in directed.items():
        pairs.setdefault(tuple(sorted((a, b))), {}).update({(a, b): int(n)})
    rows = []
    for (a, b), counts in pairs.items():
        total = sum(counts.values())
        rows.append({
            "a": a, "b": b,
            "a_name": info["name"].get(a, a), "b_name": info["name"].get(b, b),
            "a_to_b": counts.get((a, b), 0), "b_to_a": counts.get((b, a), 0),
            "errors": total,
            "rate": round(total / (support.get(a, 0) + support.get(b, 0)), 4),
            "same_family": bool(info["family"].get(a) and info["family"].get(a) == info["family"].get(b)),
            "same_script": bool(info["script"].get(a) and info["script"].get(a) == info["script"].get(b)),
        })
    return sorted(rows, key=lambda r: (-r["errors"], r["a"]))[:top]


def worst_languages(df: pd.DataFrame, info: pd.DataFrame, n: int = 10) -> list[dict]:
    labels = sorted(df["label"].unique())
    f1 = f1_score(df["label"], df["pred"], labels=labels, average=None)
    rows = []
    for code, score in sorted(zip(labels, f1), key=lambda kv: kv[1])[:n]:
        wrong = df[(df["label"] == code) & (df["pred"] != code)]["pred"].value_counts()
        rows.append({
            "code": code, "name": info["name"].get(code, code), "f1": round(float(score), 4),
            "n": int((df["label"] == code).sum()),
            "most_confused_with": wrong.index[0] if len(wrong) else None,
            "most_confused_name": info["name"].get(wrong.index[0], "") if len(wrong) else None,
        })
    return rows


def confidence_table(df: pd.DataFrame) -> dict:
    """Với mỗi ngưỡng: tỉ lệ dự đoán được giữ (confidence >= ngưỡng) và độ chính xác của chúng."""
    out = {}
    for name, g in length_groups(df).items():
        if name in [natural_label(*b) for b in NATURAL_BUCKETS]:
            continue
        correct = (g["label"] == g["pred"]).to_numpy()
        conf = g["confidence"].to_numpy()
        out[name] = {}
        for t in THRESHOLDS:
            kept = conf >= t
            out[name][str(t)] = {
                "coverage": round(float(kept.mean()), 4),
                "accuracy_kept": round(float(correct[kept].mean()), 4) if kept.any() else None,
                "accuracy_unsure": round(float(correct[~kept].mean()), 4) if (~kept).any() else None,
            }
    return out


def load_info() -> pd.DataFrame:
    info = read_split(LABELS_PATH).set_index("code")
    eda = REPORTS_DIR / "eda_processed.json"
    scripts = {r["code"]: r["script"] for r in read_json(eda)["languages"]} if eda.exists() else {}
    info["script"] = pd.Series(scripts)
    return info.fillna("")


def run_name(path: Path) -> str:
    return path.stem.removeprefix("predictions_")


def pretty_name(name: str) -> str:
    """"char_wb_1_4_100k__sgd__crop" -> "SGD, có crop" (tên file khác thì giữ nguyên)."""
    parts = name.split("__")
    if len(parts) != 3:
        return name
    return f"{parts[1].upper()}, {'có' if parts[2] == 'crop' else 'không'} crop"


def to_markdown(report: dict, figures: list[Path]) -> str:
    main = report["runs"][0]
    lines = [
        "# Phân tích lỗi theo độ dài và cặp ngôn ngữ",
        "",
        f"Cấu hình chính: `{main['name']}`. Tập test: {report['n_test']:,} đoạn đầy đủ; các nhóm "
        "\"N ký tự\" là chính các đoạn đó được cắt ngắn còn khoảng N ký tự (data/augment.py).",
        "",
        "## Độ chính xác theo độ dài",
        "",
        "| Nhóm | Số mẫu | " + " | ".join(pretty_name(r["name"]) for r in report["runs"]) + " |",
        "|---|---:|" + "---:|" * len(report["runs"]),
    ]
    for group, stats in main["by_length"].items():
        cells = " | ".join(f"{r['by_length'][group]['accuracy']:.1%} / {r['by_length'][group]['macro_f1']:.1%}"
                           for r in report["runs"])
        lines.append(f"| {group} | {stats['n']:,} | {cells} |")
    lines += ["", "Mỗi ô: accuracy / macro-F1."]

    if "by_overlap" in main:
        lines += ["", "## Ảnh hưởng của văn bản chung với train", "",
                  "Điểm trên đoạn test đầy đủ, tách theo phần văn bản chung với train (bài do bot tạo theo mẫu, "
                  "chỉ khác tên riêng – xem `data/check_leakage.py`).", "",
                  "| Nhóm | Số đoạn | " + " | ".join(pretty_name(r["name"]) for r in report["runs"]) + " |",
                  "|---|---:|" + "---:|" * len(report["runs"])]
        for group, stats in main["by_overlap"].items():
            cells = " | ".join(f"{r['by_overlap'][group]['accuracy']:.2%} / {r['by_overlap'][group]['macro_f1']:.2%}"
                               for r in report["runs"])
            lines.append(f"| {group} | {stats['n']:,} | {cells} |")
        lines += ["", "Mỗi ô: accuracy / macro-F1. Các nhóm có tỉ lệ ngôn ngữ khác nhau, nên chênh lệch giữa "
                      "các dòng không hoàn toàn do rò rỉ."]

    for bucket, title in (("full", "đoạn đầy đủ"), ("20", "đoạn 20 ký tự")):
        lines += ["", f"## Cặp ngôn ngữ hay bị nhầm – {title}", "",
                  "| Cặp | Lỗi a→b | Lỗi b→a | Tổng lỗi | Tỉ lệ | Cùng họ | Cùng hệ chữ |",
                  "|---|---:|---:|---:|---:|:---:|:---:|"]
        for p in main["pairs"][bucket]:
            lines.append(f"| {p['a_name']} (`{p['a']}`) – {p['b_name']} (`{p['b']}`) | {p['a_to_b']} | {p['b_to_a']} "
                         f"| {p['errors']} | {p['rate']:.1%} | {'✓' if p['same_family'] else ''} "
                         f"| {'✓' if p['same_script'] else ''} |")
        share_family = np.mean([p["same_family"] for p in main["pairs"][bucket]])
        lines += ["", f"Tỉ lệ = tổng lỗi giữa hai ngôn ngữ / tổng số mẫu của cả hai. "
                      f"{share_family:.0%} trong {len(main['pairs'][bucket])} cặp đứng đầu thuộc cùng họ ngôn ngữ."]

    for bucket, title in (("full", "đoạn đầy đủ"), ("20", "đoạn 20 ký tự")):
        lines += ["", f"## 10 ngôn ngữ có F1 thấp nhất – {title}", "",
                  "| Ngôn ngữ | F1 | Số mẫu | Hay bị đoán thành |", "|---|---:|---:|---|"]
        for w in main["worst"][bucket]:
            lines.append(f"| {w['name']} (`{w['code']}`) | {w['f1']:.1%} | {w['n']} "
                         f"| {w['most_confused_name']} (`{w['most_confused_with']}`) |")

    if main.get("confidence"):
        lines += ["", "## Ngưỡng độ tin cậy (`is_unsure`)", "",
                  "Mỗi ô: phần trăm dự đoán có confidence ≥ ngưỡng / độ chính xác của phần đó / độ chính xác "
                  "của phần bị gắn cờ unsure.", "",
                  "| Nhóm | " + " | ".join(f"≥ {t}" for t in THRESHOLDS) + " |",
                  "|---|" + "---|" * len(THRESHOLDS)]
        for group, row in main["confidence"].items():
            cells = []
            for t in THRESHOLDS:
                c = row[str(t)]
                acc_k = f"{c['accuracy_kept']:.1%}" if c["accuracy_kept"] is not None else "–"
                acc_u = f"{c['accuracy_unsure']:.1%}" if c["accuracy_unsure"] is not None else "–"
                cells.append(f"{c['coverage']:.0%} / {acc_k} / {acc_u}")
            lines.append(f"| {group} | " + " | ".join(cells) + " |")

    if figures:
        lines += ["", "## Hình", ""] + [f"![{p.stem}](figures/{p.name})" for p in figures]
    return "\n".join(lines) + "\n"


def _style(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot(report: dict, out_dir: Path) -> list[Path]:
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

    groups = [g for g in report["runs"][0]["by_length"] if g != "Toàn bộ test"]
    fig, ax = plt.subplots(figsize=(8.5, 4))
    x = np.arange(len(groups))
    for run, color in zip(report["runs"], COLORS):
        acc = [run["by_length"][g]["accuracy"] * 100 for g in groups]
        ax.plot(x, acc, color=color, linewidth=2, marker="o", markersize=5, label=pretty_name(run["name"]))
    ax.axvline(len(SHORT_BUCKETS) - 0.5, color=GRID, linewidth=1)
    # x theo dữ liệu, y theo tỉ lệ trục (0 = đáy, 1 = đỉnh)
    ax.text(len(SHORT_BUCKETS) - 0.42, 0.04, "đoạn đầy đủ →", fontsize=8, color=INK_2,
            transform=ax.get_xaxis_transform())
    ax.text(len(SHORT_BUCKETS) - 0.58, 0.04, "← đoạn cắt ngắn", fontsize=8, color=INK_2, ha="right",
            transform=ax.get_xaxis_transform())
    ax.set_xticks(x, groups)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Độ dài văn bản (ký tự)")
    ax.set_title("Độ chính xác giảm mạnh khi văn bản ngắn")
    _style(ax)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="lower right")
    path = out_dir / "error_by_length.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, (bucket, title) in zip(axes, (("full", "Đoạn đầy đủ"), ("20", "Đoạn 20 ký tự"))):
        pairs = report["runs"][0]["pairs"][bucket][:10][::-1]
        ax.barh([f"{p['a']} – {p['b']}" for p in pairs], [p["errors"] for p in pairs],
                height=0.7, color=COLORS[0], linewidth=0)
        for y, p in enumerate(pairs):
            ax.text(p["errors"], y, f" {p['errors']}", va="center", fontsize=8, color=INK_2)
        ax.set_title(f"10 cặp nhầm nhiều nhất – {title}")
        ax.set_xlabel("Số lỗi nhầm qua lại")
        _style(ax, grid_axis="x")
    fig.tight_layout()
    path = out_dir / "error_confused_pairs.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def main(argv=None):
    setup_console()
    parser = argparse.ArgumentParser(description="Phân tích lỗi theo độ dài và cặp ngôn ngữ")
    parser.add_argument("predictions", type=Path, nargs="+")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    info = load_info()
    shared = load_shared_fraction()
    runs = []
    for i, path in enumerate(args.predictions):
        df = read_split(path)
        df["n_chars"] = df["n_chars"].astype(int)
        run = {"name": run_name(path), "file": rel(path), "by_length": by_length(df)}
        if shared is not None:
            run["by_overlap"] = by_overlap(df, shared)
        if i == 0:
            n_test = int((df["bucket"] == "full").sum())
            run["pairs"] = {b: confused_pairs(df[df["bucket"] == b], info) for b in PAIR_BUCKETS}
            run["worst"] = {b: worst_languages(df[df["bucket"] == b], info) for b in PAIR_BUCKETS}
            if "confidence" in df.columns:
                df["confidence"] = df["confidence"].astype(float)
                run["confidence"] = confidence_table(df)
        runs.append(run)

    report = {"n_test": n_test, "runs": runs}
    figures = plot(report, args.out_dir / "figures")
    write_json(report, args.out_dir / "error_analysis.json")
    md = args.out_dir / "error_analysis.md"
    md.write_text(to_markdown(report, figures), encoding="utf-8", newline="\n")
    print(f"Đã ghi {rel(md)}, error_analysis.json và {len(figures)} hình.")


if __name__ == "__main__":
    main()
