"""Cập nhật mô hình: xử lý dữ liệu -> train -> đánh giá -> tạo version mới trong models/.

    make retrain                                        # hoặc: python -m scripts.retrain
    python -m scripts.retrain --algorithm svm --features char_wb_1_3_100k --no-augment
    python -m scripts.retrain --feedback exports/accepted_reports.csv   # file từ data.export_reports
    python -m scripts.retrain --skip-data               # dữ liệu đã có, chỉ train lại
    python -m scripts.retrain --skip-data --tune        # chọn lại tham số mô hình trên val (chậm hơn)

Các bước:
  1. Dữ liệu: data.pipeline (dùng lại data/splits/split_ids.csv nên luôn cùng một cách chia),
     gồm cả bước cắt ngắn tạo train_crops.csv và các tập đánh giá ngắn.
  2. Feedback (tùy chọn): data.feedback gộp các báo cáo đã duyệt vào train_feedback.csv.
  3. Train: `python -m ml.train` (phần của B) ghi model.joblib vào thư mục tạm.
  4. Đánh giá: `python -m ml.evaluate` (phần của B) ghi metrics.json vào cùng thư mục.
  5. Kiểm tra đủ file, ghi meta.json rồi đổi tên thư mục tạm thành models/vN: version mới chỉ
     xuất hiện khi mọi bước thành công. Version mới ở trạng thái candidate; promote qua API
     (POST /api/v1/models/promote), script này không đổi version đang chạy.

Interface ml.train / ml.evaluate cần hỗ trợ (thỏa thuận A – B):

  python -m ml.train --data-dir data/processed --out-dir <thư mục> --features <preset>
                     --algorithm <baseline|naive_bayes|svm> [--augment] [--feedback] [--tune]
      -> <thư mục>/model.joblib: sklearn Pipeline (vectorizer từ ml.features + bộ phân loại),
         có predict / predict_proba trên danh sách chuỗi đã chuẩn hóa.
         <thư mục>/train_info.json (tùy chọn): {"hyperparameter", "value", ...} được chép vào meta.json.
         --algorithm: chọn mô hình trong ml/baseline.py, ml/naive_bayes.py, ml/svm.py
         --augment  : học thêm data/processed/train_crops.csv
         --feedback : học thêm data/processed/train_feedback.csv
         --tune     : thử các giá trị tham số của mô hình, chọn theo val
  python -m ml.evaluate --model-dir <thư mục> --data-dir data/processed
      -> <thư mục>/metrics.json, tối thiểu {"accuracy", "macro_f1"} trên test.csv; nên có thêm
         "by_length" trên test_short.csv và "no_overlap" ({"accuracy", "macro_f1"}) trên các đoạn
         test không có trong overlap_with_train.csv.
"""
from __future__ import annotations

import argparse
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import sklearn

from data.common import (
    FEEDBACK_TRAIN_PATH,
    PROCESSED_DIR,
    ROOT,
    STATS_PATH,
    file_hash,
    read_json,
    rel,
    setup_console,
    write_json,
)
from ml.features import DEFAULT_PRESET, FEATURE_PRESETS, preset_params

MODELS_DIR = ROOT / "models"
# Trùng tên các file mô hình của B trong ml/. Mặc định theo kết quả trên val (experiments/results).
ALGORITHMS = ("baseline", "naive_bayes", "svm")
DEFAULT_ALGORITHM = "naive_bayes"
REQUIRED_FILES = ("model.joblib", "metrics.json")
REQUIRED_METRICS = ("accuracy", "macro_f1")
VERSION_RE = re.compile(r"^v(\d+)$")


class RetrainError(Exception):
    pass


def next_version(models_dir: Path) -> str:
    numbers = [int(m.group(1)) for p in models_dir.iterdir()
               if p.is_dir() and (m := VERSION_RE.match(p.name))] if models_dir.exists() else []
    return f"v{max(numbers, default=0) + 1}"


def run_step(name: str, args: list[str]) -> None:
    cmd = [sys.executable, "-m", *args]
    print(f"\n[{name}] {' '.join(args)}", flush=True)
    result = subprocess.run(cmd, cwd=ROOT, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    if result.stderr:
        sys.stderr.write(result.stderr)
    if result.returncode != 0:
        hint = ""
        if "NotImplementedError" in result.stderr:
            hint = (f"\n{args[0]} chưa được hiện thực. Interface cần có xem ở đầu file "
                    f"scripts/retrain.py (phần thỏa thuận A – B).")
        raise RetrainError(f"bước {name} thất bại (exit code {result.returncode}).{hint}")


def git_commit() -> str | None:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                               text=True, check=True).stdout.strip()
        return f"{sha}{'-dirty' if dirty else ''}"
    except (OSError, subprocess.CalledProcessError):
        return None


def data_version() -> dict | None:
    if not STATS_PATH.exists():
        return None
    stats = read_json(STATS_PATH)
    return {
        "split_ids_sha256": stats["split"]["split_ids_sha256"],
        "raw_md5": stats["dataset"]["raw_md5"],
        "train_sha256": stats["splits"]["train"]["sha256"],
        "test_sha256": stats["splits"]["test"]["sha256"],
    }


def hyperparameters(model_dir: Path) -> dict | None:
    """{tên: giá trị} từ train_info.json của ml.train, None nếu không có."""
    path = model_dir / "train_info.json"
    if not path.exists():
        return None
    info = read_json(path)
    return {info["hyperparameter"]: info.get("value")} if info.get("hyperparameter") else {}


def build_meta(version: str, args, metrics: dict, model_dir: Path) -> dict:
    params = preset_params(args.features)
    feedback = None
    if args.feedback:
        report = read_json(FEEDBACK_TRAIN_PATH.with_name("feedback_report.json"))
        feedback = {"source_file": args.feedback.name, "source_sha256": file_hash(args.feedback),
                    "accepted": report["accepted"], "input": report["input"]}
    return {
        "version": version,
        "status": "candidate",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "algorithm": args.algorithm,
        "features": {"preset": args.features,
                     **{k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items() if k != "dtype"}},
        "hyperparameters": hyperparameters(model_dir),
        "tune": args.tune,
        "augment": args.augment,
        "feedback": feedback,
        "data_version": data_version(),
        "metrics": {k: metrics[k] for k in REQUIRED_METRICS},
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "command": " ".join(["python -m scripts.retrain", *sys.argv[1:]]),
    }


def retrain(args) -> Path:
    if not args.skip_data:
        run_step("dữ liệu", ["data.pipeline", *(["--regenerate"] if args.regenerate_split else [])])
    if args.feedback:
        run_step("feedback", ["data.feedback", str(args.feedback)])

    args.models_dir.mkdir(parents=True, exist_ok=True)
    version = next_version(args.models_dir)
    tmp = args.models_dir / f".{version}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    try:
        run_step("train", [args.train_module, "--data-dir", rel(PROCESSED_DIR), "--out-dir", rel(tmp),
                           "--features", args.features, "--algorithm", args.algorithm,
                           *(["--augment"] if args.augment else []),
                           *(["--feedback"] if args.feedback else []),
                           *(["--tune"] if args.tune else [])])
        run_step("đánh giá", [args.eval_module, "--model-dir", rel(tmp), "--data-dir", rel(PROCESSED_DIR)])

        missing = [f for f in REQUIRED_FILES if not (tmp / f).exists()]
        if missing:
            raise RetrainError(f"thiếu file sau khi train/đánh giá: {missing}")
        metrics = read_json(tmp / "metrics.json")
        missing = [k for k in REQUIRED_METRICS if k not in metrics]
        if missing:
            raise RetrainError(f"metrics.json thiếu {missing}")

        write_json(build_meta(version, args, metrics, tmp), tmp / "meta.json")
        final = args.models_dir / version
        tmp.rename(final)
    except BaseException:
        if not args.keep_failed:
            shutil.rmtree(tmp, ignore_errors=True)
        raise
    return final


def main(argv=None) -> Path:
    setup_console()
    parser = argparse.ArgumentParser(description="Xử lý dữ liệu -> train -> đánh giá -> tạo version mới")
    parser.add_argument("--features", choices=sorted(FEATURE_PRESETS), default=DEFAULT_PRESET)
    parser.add_argument("--algorithm", choices=ALGORITHMS, default=DEFAULT_ALGORITHM)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True,
                        help="học thêm các đoạn cắt ngắn (mặc định: có)")
    parser.add_argument("--feedback", type=Path, help="CSV báo cáo đã duyệt xuất từ API")
    parser.add_argument("--tune", action="store_true", help="chọn tham số mô hình trên val (chậm hơn)")
    parser.add_argument("--skip-data", action="store_true", help="bỏ qua bước xử lý dữ liệu")
    parser.add_argument("--regenerate-split", action="store_true", help="chia lại split (ghi đè split_ids.csv)")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--train-module", default="ml.train")
    parser.add_argument("--eval-module", default="ml.evaluate")
    parser.add_argument("--keep-failed", action="store_true", help="giữ thư mục tạm khi lỗi để debug")
    args = parser.parse_args(argv)

    if args.feedback and not args.feedback.exists():
        raise SystemExit(f"Không thấy file feedback: {args.feedback}")
    try:
        final = retrain(args)
    except RetrainError as e:
        raise SystemExit(f"\nRetrain dừng: {e}\nKhông có version mới nào được tạo.") from None

    metrics = read_json(final / "meta.json")["metrics"]
    print(f"\nĐã tạo {rel(final)} (candidate): accuracy {metrics['accuracy']:.2%}, "
          f"macro-F1 {metrics['macro_f1']:.2%}")
    print(f'Đưa lên API: POST /api/v1/models/promote {{"version": "{final.name}"}} (header X-Admin-Token)')
    return final


if __name__ == "__main__":
    main()
