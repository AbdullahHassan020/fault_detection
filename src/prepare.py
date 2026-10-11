"""Task 1: load, clean, split and standardise the data."""
from pathlib import Path
import numpy as np
import pandas as pd
from load_data import load_raw

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]


def load_clean():
    """Raw CSV without the empty columns, label renamed to 'fault'."""
    df = load_raw(ROOT / "data").dropna(axis=1, how="all")
    return df.rename(columns={"Output (S)": "fault"})


def split_indices(n):
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    n_train, n_val = int(0.6 * n), int(0.2 * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


def make_features(X, feature_set):
    """Set R = 6 raw columns. Set Q = raw columns plus their squares (12 columns)."""
    if feature_set == "Q":
        return np.hstack([X, X ** 2])
    return X


def prepare(feature_set="R"):
    """Standardised train/val/test arrays. Mean and std come from the training rows only."""
    df = load_clean()
    raw = df[FEATURES].to_numpy(dtype=float)
    y = df["fault"].to_numpy(dtype=int)
    X = make_features(raw, feature_set)      # squares come from the raw values

    train_idx, val_idx, test_idx = split_indices(len(df))
    mean = X[train_idx].mean(axis=0)
    std = X[train_idx].std(axis=0)

    out = {"mean": mean, "std": std}
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        out["X_" + name] = (X[idx] - mean) / std
        out["y_" + name] = y[idx]
        out["raw_" + name] = raw[idx]
    return out


if __name__ == "__main__":
    raw = load_raw(ROOT / "data")
    print("Shape:", raw.shape)
    print("Columns:", raw.columns.tolist())
    print(raw.head(5).to_string())

    df = load_clean()
    print("\nRemoved (fully empty) columns:", [c for c in raw.columns if raw[c].isna().all()])
    print("Columns now:", df.columns.tolist())
    print("Missing values per column:")
    print(df.isna().sum())
    assert all(pd.api.types.is_numeric_dtype(df[c]) for c in FEATURES)
    assert "fault" not in FEATURES

    print("\nFault counts:")
    print(df["fault"].value_counts())

    d = prepare()
    table = pd.DataFrame({"split": ["train", "val", "test"],
                          "rows": [len(d["y_" + s]) for s in ["train", "val", "test"]],
                          "fault_fraction": [round(d["y_" + s].mean(), 4) for s in ["train", "val", "test"]]})
    print("\n", table.to_string(index=False))
    (ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)
    table.to_csv(ROOT / "results" / "tables" / "split_summary.csv", index=False)

    print("\nTrain mean after scaling:", d["X_train"].mean(axis=0).round(3))
    print("Train std after scaling:", d["X_train"].std(axis=0).round(3))
