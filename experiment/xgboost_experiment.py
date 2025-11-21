# ====================================================================
# UNIVERSAL EXPERIMENT PIPELINE – XGBoost (Custom Implementation)
# Academic / Research-Grade | Fully Automatic | Binary & Multiclass
# ====================================================================

import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from src.XGBoost import XGBoostClassifier  # Custom XGBoost của bạn

# ===========================================================================
# Scaler Factory
# ===========================================================================
def get_scaler(name: str):
    from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                       Normalizer, MaxAbsScaler, QuantileTransformer)
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
# Universal Preprocessing
# ===========================================================================
def universal_preprocessing(path: str, scaler_name="StandardScaler", apply_smote=True, random_state=42):
    print(f"\n[INFO] Loading dataset: {os.path.basename(path)}")
    df = pd.read_csv(path)

    # Auto-detect label column
    possible_labels = ["Result", "Dataset", "Class", "selector", "target", "Stage", "status", "Diagnosis"]
    label_col = next((c for c in possible_labels if c in df.columns), None)
    if label_col is None:
        raise ValueError("Không tìm thấy cột nhãn!")

    y_raw = df[label_col].copy()
    unique_vals = sorted(y_raw.dropna().unique().astype(int))

    if len(unique_vals) > 2 or label_col == "Stage":
        task = "multiclass"
        n_classes = len(unique_vals)
        label_mapping = {old: new for new, old in enumerate(unique_vals)}
        y = y_raw.map(label_mapping).astype(int).values
        print(f"[INFO] Multiclass → Labels remapped: {unique_vals} → 0..{n_classes-1}")
    else:
        task = "binary"
        n_classes = 2
        pos_label = int(y_raw.max())
        y = (y_raw == pos_label).astype(int).values
        print(f"[INFO] Binary task → Positive class = {pos_label}")

    # Drop rows with missing label
    if y_raw.isnull().any():
        df = df.dropna(subset=[label_col]).reset_index(drop=True)

    X = df.drop(columns=[label_col])
    feature_names = X.columns.tolist()

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

    # Train-test split
    stratify = y if task == "binary" or n_classes > 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=stratify
    )

    # SMOTE only for binary
    if apply_smote and task == "binary":
        X_train, y_train = SMOTE(random_state=random_state).fit_resample(X_train, y_train)
        print(f"[INFO] SMOTE applied → X_train shape: {X_train.shape}")

    print(f"[INFO] Task: {task.upper()} | Classes: {n_classes} | Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    return X_train, y_train, X_test, y_test, task, feature_names, unique_vals

# ===========================================================================
# MAIN EXPERIMENT
# ===========================================================================
if __name__ == "__main__":

    DATASET_PATH = "../data/processed/indian_liver_patient_preprocessed.csv"
    OUTPUT_CSV   = "../experiment_result/xgboost_indian_liver_patient_result.csv"

    # DATASET_PATH = "../data/processed/liver_cirrhosis_preprocessed.csv"
    # OUTPUT_CSV   = "../experiment_result/xgboost_liver_cirrhosis_result.csv"

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # Header CSV - sạch sẽ, đầy đủ, không dư thừa
    if not os.path.exists(OUTPUT_CSV):
        header = (
            "Dataset,Task,eta,max_depth,subsample,colsample_bytree,num_boost_round,best_iteration,"
            "Test_Accuracy,Test_F1_Weighted,Test_F1_Macro,"
            "Test_Precision_Weighted,Test_Recall_Weighted,"
            "Train_Time_s,Test_Time_s,Scaler,Timestamp\n"
        )
        with open(OUTPUT_CSV, "w") as f:
            f.write(header)

    # Các cấu hình XGBoost muốn thử
    XGB_CONFIGS = [
        {"eta": 0.05, "max_depth": 6,  "subsample": 0.8, "colsample_bytree": 0.8, "num_boost_round": 1000},
        {"eta": 0.1,  "max_depth": 8,  "subsample": 0.9, "colsample_bytree": 0.9, "num_boost_round": 800},
        {"eta": 0.03, "max_depth": 10, "subsample": 1.0, "colsample_bytree": 1.0, "num_boost_round": 1500},
        {"eta": 0.2,  "max_depth": 5,  "subsample": 0.7, "colsample_bytree": 0.7, "num_boost_round": 500},
    ]

    SCALERS = [None, "StandardScaler", "MinMaxScaler", "RobustScaler", "QuantileTransformer"]

    for scaler_name in SCALERS:
        X_train, y_train, X_test, y_test, task, feature_names, original_labels = universal_preprocessing(
            path=DATASET_PATH,
            scaler_name=scaler_name,
            apply_smote=True,
            random_state=42
        )

        for cfg in XGB_CONFIGS:
            print(f"\n{'='*80}")
            print(f"XGBoost | η={cfg['eta']:.3f} | depth={cfg['max_depth']} | "
                  f"sub={cfg['subsample']} | col={cfg['colsample_bytree']} | Scaler: {scaler_name or 'None'}")
            print(f"{'='*80}")

            model = XGBoostClassifier(
                eta=cfg["eta"],
                max_depth=cfg["max_depth"],
                subsample=cfg["subsample"],
                colsample_bytree=cfg["colsample_bytree"],
                num_boost_round=cfg["num_boost_round"],
                early_stopping_rounds=50,
                verbose=False
            )

            # Train
            start_train = time.time()
            model.fit(X_train, y_train, X_test, y_test)  # eval_set để early stopping
            train_time = time.time() - start_train

            # Predict
            start_test = time.time()
            y_pred = model.predict(X_test)
            test_time = time.time() - start_test

            # Chuyển lại nhãn gốc để báo cáo chính xác
            if task == "multiclass":
                y_test_orig = np.array([original_labels[i] for i in y_test])
                y_pred_orig = np.array([original_labels[i] for i in y_pred])
            else:
                y_test_orig = y_test
                y_pred_orig = y_pred

            # Metrics
            report = classification_report(y_test_orig, y_pred_orig, output_dict=True, zero_division=0)

            row = [
                os.path.basename(DATASET_PATH),
                task,
                cfg["eta"],
                cfg["max_depth"],
                cfg["subsample"],
                cfg["colsample_bytree"],
                cfg["num_boost_round"],
                getattr(model.model, "best_iteration", cfg["num_boost_round"]) + 1,  # +1 vì XGBoost đếm từ 0
                round(report["accuracy"], 4),
                round(report["weighted avg"]["f1-score"], 4),
                round(report["macro avg"]["f1-score"], 4),
                round(report["weighted avg"]["precision"], 4),
                round(report["weighted avg"]["recall"], 4),
                round(train_time, 4),
                round(test_time, 4),
                scaler_name if scaler_name else "None",
                time.strftime("%Y%m%d_%H%M%S")
            ]

            pd.DataFrame([row]).to_csv(OUTPUT_CSV, mode="a", header=False, index=False)

            print(f"SAVED | Acc: {report['accuracy']:.4f} | "
                  f"F1w: {report['weighted avg']['f1-score']:.4f} | "
                  f"Precision_w: {report['weighted avg']['precision']:.4f} | "
                  f"Recall_w: {report['weighted avg']['recall']:.4f} | "
                  f"Best iter: {row[7]}")

    print(f"\nHOÀN TẤT! Tất cả kết quả XGBoost đã được lưu tại:")
    print(f" → {OUTPUT_CSV}")
    print("Bảng kết quả sạch sẽ, đầy đủ Precision & Recall (weighted) – sẵn sàng so sánh với KNN, SVM, RF!")