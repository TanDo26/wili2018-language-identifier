"""Load splits.
Uses the official split from the data team (data/split/{train,val,test}.parquet,
columns: id, text, label) if it exists; otherwise falls back to a TEMPORARY split
(downloads WiLI-2018 and splits the train set 90/10, seed 42)."""
from pathlib import Path
import pandas as pd

SPLIT_DIR = Path("data/split")
PARTS = ("train", "val", "test")

def load_splits(split_dir=SPLIT_DIR, seed=42):
    d = Path(split_dir)
    if all((d / f"{s}.parquet").exists() for s in PARTS):
        dfs = {s: pd.read_parquet(d / f"{s}.parquet") for s in PARTS}
        data = {s: (df["text"].tolist(), df["label"].astype(str).tolist()) for s, df in dfs.items()}
        data["source"] = "official split (parquet, seed 42)"
        return data
    return _temporary_split(seed)

def _temporary_split(seed):
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split
    ds = load_dataset("MartinThoma/wili_2018", "WiLI-2018 dataset")
    names = ds["train"].features["label"].names
    X = ds["train"]["sentence"]
    y = [names[i] for i in ds["train"]["label"]]
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.1, stratify=y, random_state=seed)
    return {"train": (Xtr, ytr), "val": (Xva, yva),
            "test": (ds["test"]["sentence"], [names[i] for i in ds["test"]["label"]]),
            "source": "TEMPORARY 90/10 split of Hugging Face train set"}
