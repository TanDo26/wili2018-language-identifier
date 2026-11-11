"""ml.evaluate giả ghi metrics.json thiếu macro_f1 – retrain phải từ chối."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()
    (Path(args.model_dir) / "metrics.json").write_text(json.dumps({"accuracy": 0.95}), encoding="utf-8")


if __name__ == "__main__":
    main()
