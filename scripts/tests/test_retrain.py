import json

import pytest

from scripts.retrain import main, next_version

FAKE = ["--skip-data", "--train-module", "scripts.tests.fake_train",
        "--eval-module", "scripts.tests.fake_evaluate"]


def test_next_version_ignores_other_folders(tmp_path):
    assert next_version(tmp_path / "missing") == "v1"
    for name in ("v1", "v3", "v10x", ".v4.tmp", "notes"):
        (tmp_path / name).mkdir()
    (tmp_path / "v7").write_text("a file, not a version")
    assert next_version(tmp_path) == "v4"


def test_retrain_creates_successive_versions_with_meta(tmp_path):
    first = main([*FAKE, "--models-dir", str(tmp_path)])
    second = main([*FAKE, "--models-dir", str(tmp_path), "--no-augment"])

    assert (first.name, second.name) == ("v1", "v2")
    for version in (first, second):
        assert {p.name for p in version.iterdir()} >= {"model.joblib", "metrics.json", "meta.json"}
    meta = json.loads((second / "meta.json").read_text(encoding="utf-8"))
    assert meta["version"] == "v2"
    assert meta["status"] == "candidate"
    assert meta["metrics"] == {"accuracy": 0.95, "macro_f1": 0.94}
    assert meta["augment"] is False
    train_args = json.loads((first / "train_args.json").read_text(encoding="utf-8"))
    assert train_args["augment"] is True and train_args["feedback"] is False
    assert not list(tmp_path.glob(".*.tmp"))


def test_failed_train_leaves_no_version(tmp_path):
    # ml.train hiện vẫn là stub của B (raise NotImplementedError)
    with pytest.raises(SystemExit) as exc:
        main(["--skip-data", "--models-dir", str(tmp_path), "--eval-module", "scripts.tests.fake_evaluate"])
    assert "chưa được hiện thực" in str(exc.value)
    assert list(tmp_path.iterdir()) == []


def test_incomplete_metrics_is_rejected(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["--skip-data", "--models-dir", str(tmp_path), "--train-module", "scripts.tests.fake_train",
              "--eval-module", "scripts.tests.fake_evaluate_incomplete"])
    assert "macro_f1" in str(exc.value)
    assert list(tmp_path.iterdir()) == []
