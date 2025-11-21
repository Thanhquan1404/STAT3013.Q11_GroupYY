"""
UNIVERSAL EXPERIMENT PIPELINE
==============================

This pipeline is designed to automatically process and evaluate datasets
from two distinct medical classification domains:

1. Indian Liver Patient Dataset (Binary classification)
2. Cirrhosis Dataset (Multi-class classification, Stage 1–4)

The system automatically:
- Detects label type (binary or multi-class)
- Applies appropriate preprocessing
- Encodes categorical variables
- Handles missing values
- Applies SMOTE only for binary imbalanced tasks
- Selects scaler dynamically through a scaling factory
- Trains a custom Logistic Regression model (GD variant)
- Computes relevant evaluation metrics
- Stores experiment results into CSV

The pipeline is fully compatible with LogisticRegressionGD (custom implementation).
"""

import os
import sys
import time
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    Normalizer, MaxAbsScaler, QuantileTransformer,
    LabelEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    matthews_corrcoef, f1_score, precision_score, recall_score,
    classification_report
)
from imblearn.over_sampling import SMOTE
from src.logistic_regression import LogisticRegressionGD


# ===========================================================================
# Scaler Factory: Centralized selector for all supported scaling techniques
# ===========================================================================
def get_scaler(name: str):
    """
    Returns a scaler object based on the provided name.
    Allows flexible switching between normalization techniques.

    Supported:
        - StandardScaler
        - MinMaxScaler
        - RobustScaler
        - Normalizer
        - MaxAbsScaler
        - QuantileTransformer (Gaussian output distribution)
    """
    scalers = {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "RobustScaler": RobustScaler(),
        "Normalizer": Normalizer(),
        "MaxAbsScaler": MaxAbsScaler(),
        "QuantileTransformer": QuantileTransformer(
            output_distribution="normal", random_state=42
        ),
        None: None
    }
    if name not in scalers:
        raise ValueError(f"Unsupported scaler: {name}")
    return scalers[name]


# ===========================================================================
# UNIVERSAL PREPROCESSING MODULE
# ===========================================================================
def universal_preprocessing(
    path: str,
    scaler="StandardScaler",
    applySMOTE=True,
    random_state=42
):
    """
    Universal preprocessing engine that detects dataset type (binary or multi-class),
    handles missing values, encodes categorical variables, performs stratified splitting,
    and applies SMOTE for binary imbalance correction.
    """

    print(f"\n[INFO] Loading dataset: {os.path.basename(path)}")
    df = pd.read_csv(path)

    # --- Identify label column automatically ---
    possible_labels = ["Result", "Dataset", "Class", "selector", "target", "Stage", "status", "Diagnosis"]
    label_col = next((c for c in possible_labels if c in df.columns), None)
    if label_col is None:
        raise ValueError("Label column not found in dataset.")

    y_raw = df[label_col].copy()

    # --- Task identification logic ---
    if label_col == "Stage" or len(np.unique(y_raw)) > 2:
        task = "multiclass"
        y = y_raw.values
        n_classes = len(np.unique(y))
    else:
        task = "binary"
        n_classes = 2
        if set(y_raw.unique()) == {1, 2}:
            y = (y_raw == 2).astype(int)
        elif set(y_raw.unique()) == {0, 1}:
            y = y_raw.astype(int)
        else:
            pos = y_raw.max()
            y = (y_raw == pos).astype(int)

    # --- Remove NaN labels (if any) ---
    if y_raw.isnull().sum() > 0:
        df = df.dropna(subset=[label_col]).reset_index(drop=True)

    X = df.drop(columns=[label_col])

    # --- Encode categorical variables ---
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # --- Numerical imputation ---
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        X[num_cols] = SimpleImputer(strategy="mean").fit_transform(X[num_cols])

    # --- Scaling ---
    scaler_obj = get_scaler(scaler)
    if scaler_obj is not None:
        X = scaler_obj.fit_transform(X)

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)

    # --- Train-test split ---
    strat = y if task == "binary" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=random_state,
        stratify=strat
    )

    # --- SMOTE for binary imbalance ---
    if applySMOTE and task == "binary":
        X_train, y_train = SMOTE(random_state=random_state).fit_resample(X_train, y_train)

    print(f"[INFO] Task: {task.upper()} | Classes: {n_classes}")
    return X_train, y_train, X_test, y_test, task


# ===========================================================================
# EVALUATION MODULE
# ===========================================================================
def evaluate(y_true, y_pred, y_scores, task):
    """
    Evaluation module for both binary and multi-class classification tasks.
    Computes all relevant metrics depending on task type.
    """

    acc = accuracy_score(y_true, y_pred) * 100

    if task == "binary":
        auc_roc = roc_auc_score(y_true, y_scores) * 100
        auc_pr = average_precision_score(y_true, y_scores) * 100
        mcc = matthews_corrcoef(y_true, y_pred)
        f1v = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

        return [auc_roc, auc_pr, acc, mcc, f1v, precision, recall]

    # Multiclass metrics
    f1_macro = f1_score(y_true, y_pred, average="macro")
    return [acc, f1_macro, ]


# ===========================================================================
# MAIN EXECUTION PIPELINE
# ===========================================================================
if __name__ == "__main__":
    # DATASET_PATH = "../data/processed/indian_liver_patient_preprocessed.csv"
    # OUTPUT_CSV = "../experiment_result/logistic_regression_indian_liver_patient_result.csv"

    DATASET_PATH = "../data/processed/liver_cirrhosis_preprocessed.csv"
    OUTPUT_CSV = "../experiment_result/logistic_regression_liver_cirrhosis_result.csv"

    SCALERS = [
        "StandardScaler",
        "MinMaxScaler",
        "RobustScaler",
        "Normalizer",
        "MaxAbsScaler",
        "QuantileTransformer"
    ]

    etas = [10, 8, 5, 3, 1, 0.8, 0.5, 0.3, 0, -0.3, -0.5, -0.8, -1]

    if not os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, "w") as f:
            f.write(
                "Dataset,Task,AUCROC,AUCPR,Accuracy,MCC,F1,Precision,Recall,"
                "Time_Train,Time_Test,Eta,Scaler\n"
            )
            # f.write(
            #     "Dataset,Task,Accuracy,F1 macro,"
            #     "Time_Train,Time_Test,Eta,Scaler\n"
            # )

    for scaler in SCALERS:
        for eta in etas:

            print(f"\n=== RUNNING: Scaler={scaler}, eta={eta} ===")

            # Preprocessing
            X_train, y_train, X_test, y_test, task = universal_preprocessing(
                path=DATASET_PATH,
                scaler=scaler,
                applySMOTE=True
            )

            model = LogisticRegressionGD(eta=eta, epochs=3000, verbose=False)

            start_train = time.time()
            model.fit(X_train, y_train)
            end_train = time.time()

            y_scores = model.predict_proba(X_test).flatten()
            y_pred = model.predict(X_test).flatten()
            end_test = time.time()

            metrics = evaluate(y_test, y_pred, y_scores, task)

            row = [
                os.path.basename(DATASET_PATH),
                task
            ] + metrics + [
                end_train - start_train,
                end_test - end_train,
                eta,
                scaler
            ]

            pd.DataFrame([row]).to_csv(
                OUTPUT_CSV, mode="a", header=False, index=False
            )

            print(f"[INFO] Saved: scaler={scaler}, eta={eta}")

