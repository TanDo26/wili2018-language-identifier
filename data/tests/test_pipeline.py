import pandas as pd

from data.check_leakage import find_leaks
from data.clean import clean
from data.common import near_dup_key, normalize_text, read_lines, read_split, write_split
from data.split import make_val_split


def _df(prefix, rows):
    return pd.DataFrame({
        "id": [f"{prefix}_{i:06d}" for i in range(len(rows))],
        "text": [text for text, _ in rows],
        "label": [label for _, label in rows],
    })


def _pool(sizes):
    rows = [(f"{label} sample number {chr(97 + i % 26)}{i}", label)
            for label, n in sizes.items() for i in range(n)]
    return _df("tr", rows)


def test_normalize_text_nfc_whitespace_and_control_chars():
    assert normalize_text("Café") == "Café"
    assert normalize_text("﻿  a\t\tb\n c\x00  ") == "a b c"


def test_near_dup_key_ignores_digits_punctuation_and_case():
    assert near_dup_key("Dân số 1.234 người (2010).") == near_dup_key("dân số 5678 người, 2019")
    assert near_dup_key("Dân số") != near_dup_key("Dan so")
    assert near_dup_key("123 - !!!") == ""


def test_read_lines_splits_only_on_newline(tmp_path):
    path = tmp_path / "x.txt"
    path.write_bytes("a b\nc\r\nd\n".encode("utf-8"))
    assert read_lines(path) == ["a b", "c", "d"]


def test_csv_roundtrip_keeps_nan_label(tmp_path):
    df = pd.DataFrame({"id": ["tr_000000"], "text": ["NA"], "label": ["nan"]})
    write_split(df, tmp_path / "s.csv")
    assert read_split(tmp_path / "s.csv").equals(df)


def test_clean_removes_duplicates_conflicts_and_train_test_overlap():
    frames = {
        "train": _df("tr", [
            ("Hello world", "eng"),          # giữ
            ("hello,  world 2!", "eng"),     # gần trùng với dòng 0
            ("Bonjour le monde", "fra"),     # đã có trong test
            ("Same text", "eng"),            # test gán nhãn khác
            ("12345 ...", "eng"),            # không có chữ cái
            ("   ", "eng"),                  # rỗng
            ("Hallo Welt", "deu"),           # giữ
        ]),
        "test": _df("te", [
            ("Bonjour le monde.", "fra"),    # giữ
            ("Same text", "deu"),            # nhãn mâu thuẫn
            ("Hola mundo", "spa"),           # giữ
            ("Hola  mundo", "spa"),          # trùng trong test
        ]),
    }
    cleaned, removed, report = clean(frames)

    assert cleaned["train"]["id"].tolist() == ["tr_000000", "tr_000006"]
    assert cleaned["test"]["id"].tolist() == ["te_000000", "te_000002"]
    assert dict(zip(removed["id"], removed["reason"])) == {
        "tr_000001": "duplicate_within_split",
        "tr_000002": "overlap_with_test",
        "tr_000003": "conflicting_labels",
        "tr_000004": "no_letters",
        "tr_000005": "empty",
        "te_000001": "conflicting_labels",
        "te_000003": "duplicate_within_split",
    }
    assert report["clean"] == {"train": 2, "test": 2}
    assert report["removed_total"] == {"train": 5, "test": 2}


def test_val_split_is_stratified_and_deterministic():
    train = _pool({"aaa": 100, "bbb": 80, "ccc": 60})
    a = make_val_split(train, val_per_lang=10, seed=42)
    shuffled = make_val_split(train.sample(frac=1, random_state=0), val_per_lang=10, seed=42)
    other_seed = make_val_split(train, val_per_lang=10, seed=7)

    assert a.equals(shuffled)
    assert not a.equals(other_seed)
    val_labels = train.set_index("id").loc[a.index[a == "val"], "label"]
    assert val_labels.value_counts().to_dict() == {"aaa": 10, "bbb": 10, "ccc": 10}


def test_val_split_keeps_half_for_small_languages():
    train = _pool({"big": 100, "small": 6})
    assignment = make_val_split(train, val_per_lang=10, seed=42)
    val_labels = train.set_index("id").loc[assignment.index[assignment == "val"], "label"]
    assert val_labels.value_counts().to_dict() == {"big": 10, "small": 3}


def _splits():
    return {
        "train": _df("tr", [("Hello world", "eng"), ("Hallo Welt", "deu")]),
        "val": pd.DataFrame({"id": ["tr_000100", "tr_000101"],
                             "text": ["Good morning", "Guten Morgen"], "label": ["eng", "deu"]}),
        "test": _df("te", [("Good night", "eng"), ("Gute Nacht", "deu")]),
    }


def test_find_leaks_passes_on_disjoint_splits():
    report = find_leaks(_splits())
    assert report["passed"], report["errors"]


def test_find_leaks_detects_near_duplicate_between_train_and_test():
    frames = _splits()
    frames["test"].loc[0, "text"] = "hello, world!!"
    report = find_leaks(frames)
    assert not report["passed"]
    assert report["pairs"]["train-test"]["near"]["count"] == 1
    assert report["pairs"]["train-test"]["exact"]["count"] == 0


def test_find_leaks_detects_id_overlap_and_wrong_source():
    frames = _splits()
    frames["val"].loc[0, "id"] = "tr_000000"
    frames["test"].loc[1, "id"] = "tr_000001"
    report = find_leaks(frames)
    assert report["pairs"]["train-val"]["id"]["count"] == 1
    assert report["sources"]["test"]["wrong"] == 1
    assert not report["passed"]


def test_find_leaks_detects_mismatch_with_split_ids():
    frames = _splits()
    ids = pd.DataFrame([(i, name) for name, df in frames.items() for i in df["id"]], columns=["id", "split"])
    assert find_leaks(frames, ids)["passed"]
    ids.loc[ids["id"] == "tr_000100", "split"] = "train"
    report = find_leaks(frames, ids)
    assert report["split_ids"]["mismatched"] == 1
    assert not report["passed"]
