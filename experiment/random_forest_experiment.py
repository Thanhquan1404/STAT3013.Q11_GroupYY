# ====================================================================
# UNIVERSAL EXPERIMENT PIPELINE – RANDOM FOREST CLASSIFIER
# Academic / Research-Grade | Fully Automatic | Binary & Multiclass
# ====================================================================
"""
UNIVERSAL RANDOM FOREST EXPERIMENT PIPELINE
===========================================
• Tự động nhận diện dataset (Indian Liver vs Cirrhosis)
• Tự động phát hiện nhãn, task (binary/multiclass)
• Universal preprocessing + scaler factory
• SMOTE chỉ dùng cho binary
• Lưu toàn bộ kết quả vào 1 file CSV tổng hợp (dễ so sánh với SVM, LR, XGBoost...)
• Đo thời gian train/test chính xác
• Không hard-code cột, đường dẫn giữ nguyên như yêu cầu
"""

import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, f1_score, accuracy_score
from imblearn.over_sampling import SMOTE

# ===========================================================================
# Scaler Factory (giống hệt pipeline trước)
# ===========================================================================
def get_scaler(name: str):
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, \
        Normalizer, MaxAbsScaler, QuantileTransformer
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
# UNIVERSAL PREPROCESSING (tự động detect tất cả)
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
        raise ValueError("Không tìm thấy cột nhãn!")

    y_raw = df[label_col].copy()

    # Task detection: binary hay multiclass
    unique_vals = np.unique(y_raw.dropna())
    if len(unique_vals) > 2 or label_col == "Stage":
        task = "multiclass"
        y = y_raw.astype(int).values
        n_classes = len(unique_vals)
    else:
        task = "binary"
        n_classes = 2
        pos_label = y_raw.max()
        y = (y_raw == pos_label).astype(int).values

    # Drop missing label
    if y_raw.isnull().any():
        df = df.dropna(subset=[label_col]).reset_index(drop=True)

    X = df.drop(columns=[label_col])

    # Encode categorical
    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Impute numerical
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
    stratify = y if task == "multiclass" or n_classes == 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=stratify
    )

    # SMOTE chỉ cho binary
    if apply_smote and task == "binary":
        X_train, y_train = SMOTE(random_state=random_state).fit_resample(X_train, y_train)
        print(f"[INFO] SMOTE applied → X_train: {X_train.shape}")

    print(f"[INFO] Task: {task.upper()} | Classes: {n_classes} | Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    return X_train, y_train, X_test, y_test, task, label_col

# ===========================================================================
# MAIN EXPERIMENT LOOP – RANDOM FOREST
# ===========================================================================
if __name__ == "__main__":

    # ====================================================================
    # CẤU HÌNH CHẠY – CHỈ CẦN THAY ĐỔI 1 DÒNG Ở ĐÂY
    # ====================================================================
    DATASET_PATH = "../data/processed/icirrhosis_encoded.csv"        # Multiclass
    # DATASET_PATH = "../data/processed/liver_cleaned.csv"           # Binary

    OUTPUT_CSV = "../experiment_result/random_forest_cirrhosis_result.csv"

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # Header CSV (chỉ ghi 1 lần)
    if not os.path.exists(OUTPUT_CSV):
        header = (
            "Dataset,Task,n_estimators,max_depth,min_samples_split,min_samples_leaf,max_features,"
            "Test_Accuracy,Test_F1_Weighted,Test_F1_Macro,Train_Time_s,Test_Time_s,Scaler,Timestamp\n"
        )
        with open(OUTPUT_CSV, "w") as f:
            f.write(header)

    # Danh sách hyperparams muốn thử
    RF_CONFIGS = [
        {"n_estimators": 100, "max_depth": None,    "min_samples_split": 2,  "min_samples_leaf": 1,  "max_features": "sqrt"},
        {"n_estimators": 300, "max_depth": None,    "min_samples_split": 2,  "min_samples_leaf": 1,  "max_features": "sqrt"},
        {"n_estimators": 500, "max_depth": 20,      "min_samples_split": 5,  "min_samples_leaf": 2,  "max_features": "sqrt"},
        {"n_estimators": 300, "max_depth": None,    "min_samples_split": 2,  "min_samples_leaf": 1,  "max_features": 0.8},
    ]

    SCALERS = ["StandardScaler", "RobustScaler", "MinMaxScaler", "QuantileTransformer"]

    for scaler_name in SCALERS:
        X_train, y_train, X_test, y_test, task, label_col = universal_preprocessing(
            path=DATASET_PATH,
            scaler_name=scaler_name,
            apply_smote=True,  # Tự động bật/tắt SMOTE
            random_state=42
        )

        for cfg in RF_CONFIGS:
            print(f"\n{'='*70}")
            print(f"RUNNING RF → {cfg['n_estimators']} trees | depth={cfg['max_depth']} | "
                  f"feat={cfg['max_features']} | scaler={scaler_name}")
            print(f"{'='*70}")

            model = RandomForestClassifier(
                n_estimators=cfg["n_estimators"],
                max_depth=cfg["max_depth"],
                min_samples_split=cfg["min_samples_split"],
                min_samples_leaf=cfg["min_samples_leaf"],
                max_features=cfg["max_features"],
                bootstrap=True,
                class_weight="balanced" if task == "multiclass" else "balanced_subsample",
                n_jobs=-1,
                random_state=42
            )

            # Train
            start_train = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_train

            # Predict
            start_test = time.time()
            y_pred = model.predict(X_test)
            test_time = time.time() - start_test

            # Metrics
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            acc = report["accuracy"]
            f1_weighted = report["weighted avg"]["f1-score"]
            f1_macro = report["macro avg"]["f1-score"] if "macro avg" in report else f1_score(y_test, y_pred, average="macro")

            # Lưu kết quả
            row = [
                os.path.basename(DATASET_PATH),
                task,
                cfg["n_estimators"],
                str(cfg["max_depth"]),
                cfg["min_samples_split"],
                cfg["min_samples_leaf"],
                str(cfg["max_features"]),
                round(acc, 4),
                round(f1_weighted, 4),
                round(f1_macro, 4),
                round(train_time, 4),
                round(test_time, 4),
                scaler_name,
                time.strftime("%Y%m%d_%H%M%S")
            ]

            pd.DataFrame([row]).to_csv(OUTPUT_CSV, mode="a", header=False, index=False)

            print(f"SAVED | Acc: {acc:.4f} | F1w: {f1_weighted:.4f} | F1m: {f1_macro:.4f}")
            print(f"Time → Train: {train_time:.2f}s | Test: {test_time:.4f}s\n")

    print(f"\nHOÀN TẤT! Tất cả kết quả Random Forest đã được lưu tại:")
    print(f"   → {OUTPUT_CSV}")
    print("   Bây giờ bạn có thể so sánh trực tiếp với SVM và Logistic Regression!")