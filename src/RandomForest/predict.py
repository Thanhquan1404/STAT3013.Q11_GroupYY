# predict.py

import argparse
import os
import numpy as np
import pandas as pd
import joblib

from config import FEATURE_COLS, ENSEMBLE_MODEL_PATTERN, N_SPLITS


def load_ensemble_models():
    models = []
    for fold in range(1, N_SPLITS + 1):
        path = ENSEMBLE_MODEL_PATTERN.format(fold=fold)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing model: {path}")
        models.append(joblib.load(path))
    return models


def ensemble_predict_from_csv(input_csv: str, output_csv: str | None = None):
    df = pd.read_csv(input_csv)

    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")

    X = df[FEATURE_COLS]
    models = load_ensemble_models()

    # Collect class probabilities
    all_probs = [m.predict_proba(X) for m in models]
    mean_probs = np.mean(all_probs, axis=0)

    df["predicted_stage"] = np.argmax(mean_probs, axis=1) + 1  # since classes = 1-4

    # add probability columns
    for i in range(mean_probs.shape[1]):
        df[f"proba_stage_{i+1}"] = mean_probs[:, i]

    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Saved predictions to {output_csv}")

    return df
