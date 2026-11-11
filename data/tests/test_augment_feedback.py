import numpy as np
import pandas as pd

from data.augment import crop, make_short_eval, make_train_crops
from data.feedback import merge_feedback

TEXT = "Hôm nay trời rất đẹp và chúng tôi đi dạo quanh hồ Gươm vào buổi sáng sớm cùng gia đình."


def _df(prefix, rows):
    return pd.DataFrame({"id": [f"{prefix}_{i:06d}" for i in range(len(rows))],
                         "text": [t for t, _ in rows], "label": [y for _, y in rows]})


def test_crop_keeps_whole_words_and_min_length():
    rng = np.random.default_rng(0)
    words = set(TEXT.split())
    for length in (20, 30, 50):
        for _ in range(50):
            piece = crop(TEXT, length, rng)
            assert len(piece) >= length
            assert piece in TEXT
            assert set(piece.split()) <= words


def test_crop_without_spaces_cuts_exact_length():
    text = "東京都は日本の首都であり、世界でも有数の大都市として知られている。" * 3
    piece = crop(text, 20, np.random.default_rng(1))
    assert len(piece) == 20 and piece in text


def test_crop_short_text_and_determinism():
    assert crop("short text", 50, np.random.default_rng(0)) == "short text"
    a = [crop(TEXT, 30, np.random.default_rng(7)) for _ in range(3)]
    b = [crop(TEXT, 30, np.random.default_rng(7)) for _ in range(3)]
    assert a == b


def test_train_crops_are_substrings_of_their_source():
    # đoạn train trong WiLI dài >= 140 ký tự, nên mọi đoạn cắt (tối đa 140) đều ngắn hơn đoạn gốc
    train = _df("tr", [(TEXT + " " + TEXT, "vie"), (TEXT.upper() + " " + TEXT, "vie"),
                       ("Ein ganz anderer Satz " * 10, "deu")])
    crops = make_train_crops(train, n_per_text=3, seed=42)
    sources = train.set_index("id")
    assert len(crops) == 9
    for r in crops.itertuples():
        assert r.id.startswith(r.source_id + "#c")
        assert r.text in sources.loc[r.source_id, "text"] and r.text != sources.loc[r.source_id, "text"]
        assert r.label == sources.loc[r.source_id, "label"]
    assert crops.equals(make_train_crops(train.sample(frac=1, random_state=0), n_per_text=3, seed=42))


def test_short_eval_has_one_crop_per_text_and_length():
    df = _df("te", [(TEXT, "vie"), ("Ein ganz anderer Satz " * 10, "deu")])
    short = make_short_eval(df, lengths=(20, 50), seed=42)
    assert len(short) == 4
    assert short.groupby("length").size().to_dict() == {20: 2, 50: 2}
    assert (short["text"].str.len() >= short["length"]).all()


def test_merge_feedback_filters_and_never_touches_eval_sets():
    splits = {
        "train": _df("tr", [("Guten Morgen, wie geht es dir heute?", "deu")]),
        "val": _df("tr", [("Good morning, how are you today my friend?", "eng")]),
        "test": _df("te", [("Bonjour, comment allez-vous aujourd'hui ?", "fra")]),
    }
    feedback = pd.DataFrame({
        "report_id": ["1", "2", "3", "4", "5", "6", "7"],
        "text": [
            "Xin chào,  hôm nay bạn thế nào?",            # nhận (sau khi chuẩn hóa khoảng trắng)
            "Chào bạn",                                   # quá ngắn
            "Một câu tiếng Việt đủ dài nhưng nhãn lạ",    # nhãn không có trong WiLI
            "xin chào, hôm nay bạn thế nào!!",            # gần trùng với báo cáo 1
            "bonjour comment allez vous aujourd hui",     # gần trùng với test -> rò rỉ
            "Guten Morgen, wie geht es dir heute?",       # đã có trong train
            "12345 67890 !!! ??? 2024-11-02",              # không có chữ cái
        ],
        "label": ["vie", "vie", "xyz", "vie", "fra", "deu", "eng"],
    })
    accepted, removed, report = merge_feedback(feedback, splits, {"vie", "fra", "deu", "eng"})

    assert accepted.to_dict("records") == [
        {"id": "fb_1", "text": "Xin chào, hôm nay bạn thế nào?", "label": "vie", "report_id": "1"}]
    assert dict(zip(removed["id"], removed["reason"])) == {
        "fb_2": "too_short", "fb_3": "unknown_label", "fb_4": "duplicate",
        "fb_5": "overlap_val_test", "fb_6": "already_in_train", "fb_7": "no_letters"}
    assert report["accepted"] == 1 and report["input"] == 7
