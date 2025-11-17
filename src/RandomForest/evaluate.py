# evaluate.py

import argparse
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from config import FEATURE_COLS, TARGET_COL, TEST_CSV_PATH
from predict import load_ensemble_models


def evaluate_ensemble(test_csv: str):
    df = pd.read_csv(test_csv)

    missing_cols = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    X = df[FEATURE_COLS]
    y_true = df[TARGET_COL].values

    models = load_ensemble_models()

    probs = np.array([m.predict_proba(X) for m in models])
    mean_probs = probs.mean(axis=0)
    y_pred = np.argmax(mean_probs, axis=1) + 1

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    print("=== Multiclass Ensemble Evaluation (Stage 1–4) ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-macro: {f1:.4f}\n")

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    return {"accuracy": acc, "f1_macro": f1}
