# ====================================================================
# UNIVERSAL EXPERIMENT PIPELINE – SUPPORT VECTOR MACHINE (Custom SVM)
# Academic / Research-Grade | Fully Automatic | Binary & Multiclass
# ====================================================================
"""
UNIVERSAL SVM EXPERIMENT PIPELINE
=================================
• Tự động nhận diện dataset (Indian Liver vs Cirrhosis)
• Tự động phát hiện nhãn + task (binary/multiclass)
• Universal preprocessing + 7 scalers
• SMOTE chỉ dùng cho binary
• Grid đầy đủ: linear, rbf, poly + C + gamma + degree
• Đo thời gian chính xác, lưu CSV chuẩn để so sánh với KNN/RF/Logistic
"""

import os
import sys
import time
import numpy as np
import pandas as pd

# Thêm project root vào path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import SMOTE
from src.svm import SVMClassifier

# ===========================================================================
# Scaler Factory (đồng bộ với KNN)
# ===========================================================================
def get_scaler(name: str):
    from sklearn.preprocessing import (
        StandardScaler, MinMaxScaler, RobustScaler,
        Normalizer, MaxAbsScaler, QuantileTransformer
    )
    scalers = {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "RobustScaler": RobustScaler(),
        "Normalizer": Normalizer(),
        "MaxAbsScaler": MaxAbsScaler(),
        "QuantileTransformer": QuantileTransformer(output_distribution="normal", random_state=42),
        None: None
    }
    return scalers.get(name, None)

# ===========================================================================
# UNIVERSAL PREPROCESSING (không leakage, chuẩn research)
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

    # Task detection
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

    # Drop missing label rows
    df = df.dropna(subset=[label_col]).reset_index(drop=True)

    X = df.drop(columns=[label_col])

    # Encode categorical
    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str).fillna("Missing"))

    # Impute numerical
    num_cols = X.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        imputer = SimpleImputer(strategy="mean")
        X[num_cols] = imputer.fit_transform(X[num_cols])

    # Scaling
    scaler = get_scaler(scaler_name)
    if scaler is not None:
        X = scaler.fit_transform(X)

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)

    # Stratified split
    stratify = y if n_classes >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=stratify
    )

    # SMOTE chỉ cho binary
    smote_status = "OFF"
    if apply_smote and task == "binary":
        try:
            sm = SMOTE(random_state=random_state)
            X_train, y_train = sm.fit_resample(X_train, y_train)
            smote_status = "ON"
            print(f"[INFO] SMOTE applied → X_train shape: {X_train.shape}")
        except Exception as e:
            print(f"[WARNING] SMOTE skipped: {e}")

    print(f"[INFO] Task: {task.upper()} | Classes: {n_classes} | Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    return X_train, y_train, X_test, y_test, task, smote_status


# ===========================================================================
# MAIN EXPERIMENT LOOP – SVM
# ===========================================================================
if __name__ == "__main__":

    # === CHỈ CẦN ĐỔI 2 DÒNG DƯỚI ĐÂY ĐỂ CHẠY DATASET KHÁC ===
    DATASET_PATH = "../data/processed/indian_liver_patient_preprocessed.csv"
    # DATASET_PATH = "../data/processed/liver_cirrhosis_preprocessed.csv"

    OUTPUT_CSV = "../experiment_result/svm_indian_liver_patient_result.csv"
    # OUTPUT_CSV = "../experiment_result/svm_liver_cirrhosis_result.csv"

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # Header CSV
    if not os.path.exists(OUTPUT_CSV):
        header = (
            "Dataset,Task,Kernel,C,Gamma,Degree,Best_CV_F1w,Test_Accuracy,Test_F1_Weighted,Test_F1_Macro,"
            "Train_Time_s,Test_Time_s,Scaler,SMOTE,Timestamp\n"
        )
        with open(OUTPUT_CSV, "w", encoding="utf-8") as f:
            f.write(header)

    # Scalers
    SCALERS = [None, "StandardScaler", "MinMaxScaler", "RobustScaler", "Normalizer", "MaxAbsScaler", "QuantileTransformer"]

    # Grid SVM
    SVM_PARAM_GRIDS = [
        {"kernel": "linear", "C": 0.1},
        {"kernel": "linear", "C": 1.0},
        {"kernel": "linear", "C": 10.0},

        {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
        {"kernel": "rbf", "C": 10.0, "gamma": "scale"},
        {"kernel": "rbf", "C": 100.0, "gamma": "scale"},
        {"kernel": "rbf", "C": 10.0, "gamma": 0.1},
        {"kernel": "rbf", "C": 10.0, "gamma": 1.0},

        {"kernel": "poly", "C": 1.0, "degree": 2, "gamma": "scale"},
        {"kernel": "poly", "C": 10.0, "degree": 2, "gamma": "scale"},
        {"kernel": "poly", "C": 10.0, "degree": 3, "gamma": "scale"},
        {"kernel": "poly", "C": 100.0, "degree": 3, "gamma": "scale"},
    ]

    total_runs = len(SCALERS) * len(SVM_PARAM_GRIDS)
    current_run = 0

    for scaler_name in SCALERS:
        X_train, y_train, X_test, y_test, task, smote_status = universal_preprocessing(
            path=DATASET_PATH,
            scaler_name=scaler_name,
            apply_smote=True,
            random_state=42
        )

        for params in SVM_PARAM_GRIDS:
            current_run += 1
            kernel = params["kernel"]
            C = params.get("C", 1.0)
            gamma = params.get("gamma", "-")
            degree = params.get("degree", "-")

            # Hiển thị đẹp, không lỗi khi None
            scaler_disp = scaler_name if scaler_name else "None"
            gamma_disp = gamma if gamma != "scale" else "scale"
            degree_disp = str(degree) if degree != "-" else "-"

            print(f"\n{'=' * 90}")
            print(f"RUN [{current_run:>3}/{total_runs}] | "
                  f"Scaler: {scaler_disp:<18} | Kernel: {kernel:<6} | "
                  f"C: {C:<8} | Gamma: {gamma_disp:<8} | Degree: {degree_disp}")
            print(f"{'=' * 90}")

            # Tạo model
            model = SVMClassifier(**params)

            # 5-fold CV (dùng F1-weighted thay accuracy → chuẩn research)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = []
            for train_idx, val_idx in cv.split(X_train, y_train):
                model.fit(X_train[train_idx], y_train[train_idx])
                pred = model.predict(X_train[val_idx])
                cv_scores.append(f1_score(y_train[val_idx], pred, average="weighted"))

            best_cv_f1 = np.mean(cv_scores)

            # Train full
            t0 = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - t0

            # Predict
            t0 = time.time()
            y_pred = model.predict(X_test)
            test_time = time.time() - t0

            # Metrics
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            test_acc = report["accuracy"]
            test_f1w = report["weighted avg"]["f1-score"]
            test_f1m = report.get("macro avg", {}).get("f1-score", f1_score(y_test, y_pred, average="macro"))

            # Lưu kết quả
            row = [
                os.path.basename(DATASET_PATH),
                task,
                kernel,
                C,
                gamma_disp,
                degree_disp,
                round(best_cv_f1, 4),
                round(test_acc, 4),
                round(test_f1w, 4),
                round(test_f1m, 4),
                round(train_time, 4),
                round(test_time, 4),
                scaler_disp,
                smote_status,
                time.strftime("%Y%m%d_%H%M%S")
            ]

            pd.DataFrame([row]).to_csv(OUTPUT_CSV, mode="a", header=False, index=False)

            print(f"SAVED | CV F1w: {best_cv_f1:.4f} → Test Acc: {test_acc:.4f} | F1w: {test_f1w:.4f} | Time: {train_time+test_time:.3f}s")

    print(f"\nHOÀN TẤT SVM EXPERIMENT!")
    print(f"Kết quả đã lưu tại: {OUTPUT_CSV}")
    print("Bây giờ bạn có bảng so sánh đầy đủ: SVM vs KNN vs RF vs Logistic")