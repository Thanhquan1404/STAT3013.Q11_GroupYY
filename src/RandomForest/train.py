# train.py

import os
import numpy as np
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score
)

from dataset import load_dataset
from model import build_model
from config import N_SPLITS, RANDOM_STATE, MODEL_PATH, ENSEMBLE_MODEL_PATTERN


def cross_validate_and_save_folds(X, y):
    """
    6-fold stratified CV for 4-class Stage prediction.
    """
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    print("Unique Stage labels:", sorted(np.unique(y)))

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_results = {"accuracy": [], "f1_macro": []}

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = build_model()
        model.fit(X_train, y_train)

        model_path = ENSEMBLE_MODEL_PATTERN.format(fold=fold_idx)
        joblib.dump(model, model_path)
        print(f"\nSaved fold {fold_idx} model to: {model_path}")

        # Evaluate
        y_pred = model.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="macro")

        fold_results["accuracy"].append(acc)
        fold_results["f1_macro"].append(f1)

        print(f"Fold {fold_idx}/{N_SPLITS}")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1-macro: {f1:.4f}")

    print("\n=== Cross-validation summary ===")
    for metric, values in fold_results.items():
        values = np.array(values)
        print(
            f"{metric:10s}: mean={values.mean():.4f}  "
            f"std={values.std():.4f}  "
            f"min={values.min():.4f}  max={values.max():.4f}"
        )


def main():
    X, y = load_dataset()
    print(f"Loaded dataset: X={X.shape}, y={y.shape}")
    print("Stage values:", sorted(np.unique(y)))

    print("Starting 6-fold CV...")
    cross_validate_and_save_folds(X, y)


if __name__ == "__main__":
    main()
