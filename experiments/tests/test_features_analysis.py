import pandas as pd
import pytest

from experiments.error_analysis import by_length, confidence_table, confused_pairs, worst_languages
from ml.features import DEFAULT_PRESET, FEATURE_PRESETS, build_vectorizer


def test_build_vectorizer_presets_and_overrides():
    vec = build_vectorizer()
    assert vec.get_params()["analyzer"] == FEATURE_PRESETS[DEFAULT_PRESET]["analyzer"]
    assert vec.strip_accents is None and vec.sublinear_tf
    assert build_vectorizer("char_1_3_100k", max_features=50).max_features == 50
    with pytest.raises(ValueError):
        build_vectorizer("no_such_preset")


def test_vectorizer_keeps_diacritics_apart():
    vec = build_vectorizer(ngram_range=(1, 1))
    vec.fit(["ma má mà mả mã mạ"])
    assert {"á", "à", "ả", "ã", "ạ"} <= set(vec.vocabulary_)


def _preds():
    rows = []
    # đoạn đầy đủ: hbs/srp nhầm qua lại, eng đúng hết
    rows += [("full", 300, "hrv", "hrv")] * 8 + [("full", 300, "hrv", "srp")] * 2
    rows += [("full", 600, "srp", "srp")] * 7 + [("full", 600, "srp", "hrv")] * 3
    rows += [("full", 1500, "eng", "eng")] * 10
    # 20 ký tự: kém hơn
    rows += [("20", 22, "hrv", "srp")] * 5 + [("20", 22, "hrv", "hrv")] * 5
    rows += [("20", 21, "eng", "eng")] * 10
    df = pd.DataFrame(rows, columns=["bucket", "n_chars", "label", "pred"])
    df["confidence"] = [0.95 if lab == pred else 0.3 for lab, pred in zip(df["label"], df["pred"])]
    return df


INFO = pd.DataFrame({"name": {"hrv": "Croatian", "srp": "Serbian", "eng": "English"},
                     "family": {"hrv": "Indo-European", "srp": "Indo-European", "eng": "Indo-European"},
                     "script": {"hrv": "Latin", "srp": "Cyrillic", "eng": "Latin"}})


def test_confused_pairs_merges_both_directions():
    pairs = confused_pairs(_preds().query("bucket == 'full'"), INFO)
    assert len(pairs) == 1
    p = pairs[0]
    assert (p["a"], p["b"], p["a_to_b"], p["b_to_a"], p["errors"]) == ("hrv", "srp", 2, 3, 5)
    assert p["rate"] == 0.25 and p["same_family"] and not p["same_script"]


def test_by_length_groups_short_and_natural_lengths():
    out = by_length(_preds())
    assert out["20 ký tự"] == {"n": 20, "accuracy": 0.75, "macro_f1": out["20 ký tự"]["macro_f1"]}
    assert out["200–499"]["n"] == 10 and out["500–999"]["n"] == 10 and out["1000+"]["n"] == 10
    assert out["Toàn bộ test"]["accuracy"] == round(25 / 30, 4)
    assert "30 ký tự" not in out and "140–199" not in out  # nhóm rỗng bị bỏ qua


def test_worst_languages_and_confidence():
    worst = worst_languages(_preds().query("bucket == 'full'"), INFO, n=1)
    assert worst[0]["code"] in {"hrv", "srp"} and worst[0]["most_confused_with"] in {"hrv", "srp"}
    conf = confidence_table(_preds())["20 ký tự"]["0.4"]
    assert conf == {"coverage": 0.75, "accuracy_kept": 1.0, "accuracy_unsure": 0.0}
