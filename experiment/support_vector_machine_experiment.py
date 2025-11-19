# ====================================================================
# UNIVERSAL EXPERIMENT PIPELINE – SUPPORT VECTOR MACHINE (Custom SVM)
# Academic / Research-Grade Version | Fully Automatic | Reproducible
# ====================================================================
"""
UNIVERSAL SVM EXPERIMENT PIPELINE
=================================
Automatically works with:
• Indian Liver Patient Dataset (binary)
• Cirrhosis Dataset (multiclass Stage 1–4)
Features:
• Auto-detect label column & task type
• Universal preprocessing (impute + encode + scale)
• SMOTE only for binary imbalanced tasks
• Scaler factory (Standard, MinMax, Robust, etc.)
• Full GridSearch over kernel, C, gamma, degree
• Saves all results to a single CSV summary (append mode)
• Compatible with custom SVMClassifier from src.svmClass
"""

import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, \
    Normalizer, MaxAbsScaler, QuantileTransformer, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, f1_score, accuracy_score

from src.svmClass import SVMClassifier

# ===========================================================================
# Scaler Factory
# ===========================================================================
def get_scaler(name: str):
    scalers = {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "RobustScaler": RobustScaler(),
        "Normalizer": Normalizer(),
        "MaxAbsScaler": MaxAbsScaler(),
        "QuantileTransformer": QuantileTransformer(output_distribution="normal", random_state=42),
        None: None
    }
    if name not in scalers:
        raise ValueError(f"Unsupported scaler: {name}")
    return scalers[name]

# ===========================================================================
# UNIVERSAL PREPROCESSING (Auto-detect everything)
# ===========================================================================
def universal_preprocessing(
    path: str,
    scaler_name="StandardScaler",
    apply_smote=True,
    random_state=42
):
    print(f"\n[INFO] Loading dataset: {os.path.basename(path)}")
    df = pd.read_csv(path)

    # Auto-detect label column
    possible_labels = ["Result", "Dataset", "Class", "selector", "target", "Stage", "status", "Diagnosis"]
    label_col = next((c for c in possible_labels if c in df.columns), None)
    if label_col is None:
        raise ValueError("Cannot detect label column!")

    y_raw = df[label_col].copy()

    # Task detection
    unique_labels = np.unique(y_raw.dropna())
    if label_col == "Stage" or len(unique_labels) > 2:
        task = "multiclass"
        y = y_raw.astype(int).values
        n_classes = len(unique_labels)
    else:
        task = "binary"
        n_classes = 2
        pos_label = y_raw.max()
        y = (y_raw == pos_label).astype(int).values

    # Drop rows with missing label
    if y_raw.isnull().sum() > 0:
        df = df.dropna(subset=[label_col]).reset_index(drop=True)

    X = df.drop(columns=[label_col])

    # Encode categorical features
    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Impute numerical features
    num_cols = X.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        X[num_cols] = SimpleImputer(strategy="mean").fit_transform(X[num_cols])

    # Scaling
    scaler = get_scaler(scaler_name)
    if scaler is not None:
        X = scaler.fit_transform(X)

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)

    # Stratified split
    stratify = y if task == "binary" or n_classes > 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=stratify
    )

    # Apply SMOTE only for binary tasks
    if apply_smote and task == "binary":
        X_train, y_train = SMOTE(random_state=random_state).fit_resample(X_train, y_train)
        print(f"[INFO] SMOTE applied → X_train shape: {X_train.shape}")

    print(f"[INFO] Task: {task.upper()} | Classes: {n_classes} | Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test, task, label_col

# ===========================================================================
# MAIN EXPERIMENT LOOP
# ===========================================================================
if __name__ == "__main__":

    # ====================================================================
    # CẤU HÌNH CHẠY (bạn chỉ cần thay đổi 1 trong 2 dòng dưới)
    # ====================================================================
    DATASET_PATH = "../data/processed/icirrhosis_encoded.csv"        # ← Multiclass
    # DATASET_PATH = "../data/processed/liver_cleaned.csv"           # ← Binary

    OUTPUT_CSV = "../experiment_result/support_vector_machine_cirrhosis_result.csv"

    # Tạo thư mục nếu chưa có
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # Khởi tạo file CSV + header (chỉ lần đầu)
    if not os.path.exists(OUTPUT_CSV):
        header = (
            "Dataset,Task,Kernel,C,Gamma,Degree,Best_CV_F1w,Test_F1w,Test_Acc,"
            "Train_Time_s,Test_Time_s,Scaler,Timestamp\n"
        )
        with open(OUTPUT_CSV, "w") as f:
            f.write(header)

    # Danh sách scaler muốn thử
    SCALERS = [
        "StandardScaler",
        "RobustScaler",
        "MinMaxScaler",
        "QuantileTransformer"
    ]

    # Grid cấu hình cho từng kernel
    PARAM_GRIDS = [
        # Linear
        {"model__kernel": ["linear"], "model__C": [0.1, 1, 10, 100, 1000]},
        # RBF
        {"model__kernel": ["rbf"], "model__C": [1, 10, 50, 100, 200],
         "model__gamma": ["scale", "auto", 0.01, 0.1, 0.5, 1.0]},
        # Polynomial
        {"model__kernel": ["poly"], "model__C": [1, 10, 100],
         "model__degree": [2, 3], "model__gamma": ["scale", "auto", 0.1, 1.0]}
    ]

    for scaler_name in SCALERS:
        X_train, y_train, X_test, y_test, task, label_col = universal_preprocessing(
            path=DATASET_PATH,
            scaler_name=scaler_name,
            apply_smote=True,  # Chỉ có hiệu lực với binary
            random_state=42
        )

        for param_grid in PARAM_GRIDS:
            print(f"\n{'='*60}")
            print(f"RUNNING → Scaler: {scaler_name} | Grid: {param_grid['model__kernel'][0]}")
            print(f"{'='*60}")

            # Pipeline chung
            svm_model = SVMClassifier(random_state=42)
            pipeline = ImbPipeline(steps=[
                ("model", svm_model)
            ])

            # GridSearch với StratifiedKFold
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=cv,
                scoring="f1_weighted",
                n_jobs=-1,
                verbose=0
            )

            # Đo thời gian train
            start_train = time.time()
            grid_search.fit(X_train, y_train)
            train_time = time.time() - start_train

            # Best model predict
            best_model = grid_search.best_estimator_
            start_test = time.time()
            y_pred = best_model.predict(X_test)
            test_time = time.time() - start_test

            # Metrics
            report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            test_f1w = report_dict["weighted avg"]["f1-score"]
            test_acc = report_dict["accuracy"]

            best_params = grid_search.best_params_
            kernel = best_params["model__kernel"]
            C = best_params.get("model__C", "-")
            gamma = best_params.get("model__gamma", "-")
            degree = best_params.get("model__degree", "-")

            # Lưu kết quả
            row = [
                os.path.basename(DATASET_PATH),
                task,
                kernel,
                C,
                gamma if gamma != "-" else "-",
                degree if degree != "-" else "-",
                round(grid_search.best_score_, 4),
                round(test_f1w, 4),
                round(test_acc, 4),
                round(train_time, 4),
                round(test_time, 4),
                scaler_name,
                time.strftime("%Y%m%d_%H%M%S")
            ]

            pd.DataFrame([row]).to_csv(OUTPUT_CSV, mode="a", header=False, index=False)

            print(f"SAVED | Kernel: {kernel} | C: {C} | γ: {gamma} | Deg: {degree}")
            print(f"Best CV F1w: {grid_search.best_score_:.4f} → Test F1w: {test_f1w:.4f} | Acc: {test_acc:.4f}")
            print(f"Time → Train: {train_time:.2f}s | Test: {test_time:.4f}s\n")

    print(f"\nHOÀN TẤT! Tất cả kết quả đã được lưu vào:\n   → {OUTPUT_CSV}")