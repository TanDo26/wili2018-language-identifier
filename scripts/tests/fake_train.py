"""ml.train giả cho test retrain: nhận đúng các tham số của interface và ghi model.joblib."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--feedback", action="store_true")
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()
    out = Path(args.out_dir)
    (out / "model.joblib").write_bytes(b"fake model")
    (out / "train_args.json").write_text(json.dumps(vars(args)), encoding="utf-8")
    (out / "train_info.json").write_text(json.dumps({"hyperparameter": "alpha", "value": 0.01}), encoding="utf-8")


if __name__ == "__main__":
    main()
