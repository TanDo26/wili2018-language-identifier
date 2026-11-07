"""ml.evaluate giả cho test retrain: ghi metrics.json tối thiểu theo interface."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()
    metrics = {"accuracy": 0.95, "macro_f1": 0.94}
    (Path(args.model_dir) / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


if __name__ == "__main__":
    main()
