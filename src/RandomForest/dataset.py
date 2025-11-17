# dataset.py

import pandas as pd
from config import CSV_PATH, TARGET_COL, FEATURE_COLS


def load_dataset(csv_path: str = CSV_PATH):
    """
    Load the preprocessed CSV and return X (features) and y (target).
    """
    df = pd.read_csv(csv_path)

    # Basic sanity check
    missing_cols = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in CSV: {missing_cols}")

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]     # <-- keep as 1 / 2, no ==2 mapping here

    return X, y
