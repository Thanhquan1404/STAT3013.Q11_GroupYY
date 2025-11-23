# SVM_experiment.py
import os
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve, auc)
import warnings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

# Import class SVM đã fix calibration
from models.Support_Vector_Machine import UnifiedSVMClassifier


def SVM_experiment(
    output_file_name,
    train_path,
    test_path,
    output_dir,
    label_col='Dataset',
    index_col=None,
    scalers=None,
    random_state=42,
    multiclass=False  # Tự động detect, nhưng vẫn giữ tham số để override nếu cần
):
    """
    Thí nghiệm SVM - Hỗ trợ cả Binary và Multiclass
    """
    os.makedirs(output_dir, exist_ok=True)

    # Đọc dữ liệu
    train_df = pd.read_csv(train_path, index_col=index_col)
    test_df  = pd.read_csv(test_path, index_col=index_col)

    X_train = train_df.drop(columns=[label_col])
    y_train = train_df[label_col]
    X_test  = test_df.drop(columns=[label_col])
    y_test  = test_df[label_col]

    # Tự động phát hiện multiclass
    n_classes = len(np.unique(y_train))
    is_multiclass = n_classes > 2
    if multiclass:
        is_multiclass = True

    print(f"Phát hiện: {'Multiclass' if is_multiclass else 'Binary'} classification ({n_classes} classes)")

    # Chuẩn hóa nhãn về 0,1,2,... nếu cần (đặc biệt với Stage 1,2,3,4)
    if y_train.min() >= 1:
        y_train = y_train - y_train.min()
        y_test = y_test - y_test.min()

    classes = sorted(np.unique(y_train))
    class_names = [f"Class {c}" for c in classes]

    # Danh sách scaler
    if scalers is None:
        scalers = {
            'NoScaler': None,
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'RobustScaler': RobustScaler(),
            'Normalizer': Normalizer()
        }

    # Danh sách model
    models = {
        'LinearSVC': UnifiedSVMClassifier(svm_type="linear_svc", C=1.0, calibration=True, random_state=random_state),
        'SGD_Hinge': UnifiedSVMClassifier(svm_type="sgd", loss="hinge", alpha=0.0001, calibration=True, random_state=random_state),
        'SGD_SqHinge': UnifiedSVMClassifier(svm_type="sgd", loss="squared_hinge", alpha=0.0001, calibration=True, random_state=random_state),
    }

    results = []

    for scaler_name, scaler in scalers.items():
        if scaler is not None:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled  = scaler.transform(X_test)
        else:
            X_train_scaled = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
            X_test_scaled  = X_test.values if isinstance(X_test, pd.DataFrame) else X_test

        for model_name, model in models.items():
            print(f"[{scaler_name} + {model_name}] Training...")

            model_dir = os.path.join(output_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)

            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)

            # === Accuracy luôn có ===
            acc = accuracy_score(y_test, y_pred)

            # === Metrics theo loại nhiệm vụ ===
            if is_multiclass:
                pre = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1  = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                # ROC-AUC: One-vs-Rest
                try:
                    roc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
                except:
                    roc = np.nan
            else:
                pre = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1  = f1_score(y_test, y_pred, zero_division=0)
                roc = roc_auc_score(y_test, y_proba[:, 1])

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = (None, None, None, None)
            if not is_multiclass and cm.size == 4:
                tn, fp, fn, tp = cm.ravel()

            results.append({
                'Scaler': scaler_name,
                'Model': model_name,
                'ACC': round(acc, 4),
                'Precision': round(pre, 4),
                'Recall': round(rec, 4),
                'F1': round(f1, 4),
                'ROC-AUC': round(roc, 4) if not np.isnan(roc) else 'N/A',
                'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn
            })

            # === Vẽ ROC Curve ===
            plt.figure(figsize=(8, 6))
            if is_multiclass:
                # One-vs-Rest ROC
                from sklearn.preprocessing import label_binarize
                y_test_bin = label_binarize(y_test, classes=classes)
                for i, class_name in enumerate(class_names):
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                    roc_auc_val = auc(fpr, tpr)
                    plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc_val:.3f})')
                plt.title(f'ROC Curve (One-vs-Rest)\n{scaler_name} + {model_name}')
            else:
                fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                roc_auc_val = auc(fpr, tpr)
                plt.plot(fpr, tpr, color='darkorange', lw=2,
                         label=f'ROC curve (AUC = {roc_auc_val:.3f})')
                plt.plot([0, 1], [0, 1], 'k--', lw=2)

            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc="lower right")
            roc_path = os.path.join(model_dir, f"ROC_{scaler_name}_{model_name}.png")
            plt.savefig(roc_path, dpi=300, bbox_inches='tight')
            plt.close()

            # === Vẽ Confusion Matrix ===
            plt.figure(figsize=(7, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names,
                        yticklabels=class_names)
            plt.title(f'Confusion Matrix\n{scaler_name} + {model_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            cm_path = os.path.join(model_dir, f"CM_{scaler_name}_{model_name}.png")
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.close()

    # === Lưu kết quả ===
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='ACC', ascending=False).reset_index(drop=True)
    csv_path = os.path.join(output_dir, f"{output_file_name}.csv")
    results_df.to_csv(csv_path, index=False)

    print(f"\nHOÀN THÀNH THÍ NGHIỆM SVM")
    print(f"→ Loại: {'Multiclass' if is_multiclass else 'Binary'} ({n_classes} lớp)")
    print(f"→ Kết quả lưu tại: {output_dir}")
    print(f"→ Best Model: {results_df.iloc[0]['Model']} + {results_df.iloc[0]['Scaler']} | ACC = {results_df.iloc[0]['ACC']}")

    return results_df

# ====================== CHẠY THỬ ======================
if __name__ == "__main__":
    # SVM for indian liver disease K_Fold = 5
    # SVM_experiment(
    #     output_file_name="indian_liver_patient_k_fold_01_experiment_result",
    #     train_path="../Data/KFold_data/indian_liver_patient_train_k_fold_01.csv",
    #     test_path="../Data/KFold_data/indian_liver_patient_test_k_fold_01.csv",
    #     output_dir="../experiment_result/indian_liver_patient/SVM/K_Fold_01",
    #     label_col="Result",   # bạn dùng "Result" chứ không phải "Dataset"
    #     random_state=42
    # )
    # SVM_experiment(
    #     output_file_name="indian_liver_patient_k_fold_02_experiment_result",
    #     train_path="../Data/KFold_data/indian_liver_patient_train_k_fold_02.csv",
    #     test_path="../Data/KFold_data/indian_liver_patient_test_k_fold_02.csv",
    #     output_dir="../experiment_result/indian_liver_patient/SVM/K_Fold_02",
    #     label_col="Result",   # bạn dùng "Result" chứ không phải "Dataset"
    #     random_state=42
    # )
    # SVM_experiment(
    #     output_file_name="indian_liver_patient_k_fold_03_experiment_result",
    #     train_path="../Data/KFold_data/indian_liver_patient_train_k_fold_03.csv",
    #     test_path="../Data/KFold_data/indian_liver_patient_test_k_fold_03.csv",
    #     output_dir="../experiment_result/indian_liver_patient/SVM/K_Fold_03",
    #     label_col="Result",   # bạn dùng "Result" chứ không phải "Dataset"
    #     random_state=42
    # )
    # SVM_experiment(
    #     output_file_name="indian_liver_patient_k_fold_04_experiment_result",
    #     train_path="../Data/KFold_data/indian_liver_patient_train_k_fold_04.csv",
    #     test_path="../Data/KFold_data/indian_liver_patient_test_k_fold_04.csv",
    #     output_dir="../experiment_result/indian_liver_patient/SVM/K_Fold_04",
    #     label_col="Result",   # bạn dùng "Result" chứ không phải "Dataset"
    #     random_state=42
    # )
    # SVM_experiment(
    #     output_file_name="indian_liver_patient_k_fold_05_experiment_result",
    #     train_path="../Data/KFold_data/indian_liver_patient_train_k_fold_05.csv",
    #     test_path="../Data/KFold_data/indian_liver_patient_test_k_fold_05.csv",
    #     output_dir="../experiment_result/indian_liver_patient/SVM/K_Fold_05",
    #     label_col="Result",   # bạn dùng "Result" chứ không phải "Dataset"
    #     random_state=42
    # )
    # SVM for indian liver disease K_Fold = 5 and SMOTE
    # SVM_experiment(
    #     output_file_name="indian_liver_patient_k_fold_01_SMOTE_experiment_result",
    #     train_path="../Data/data_apply_SMOTE/KFold_data/indian_liver_patient_train_k_fold_01_SMOTE.csv",
    #     test_path="../Data/KFold_data/indian_liver_patient_test_k_fold_01.csv",
    #     output_dir="../experiment_result/indian_liver_patient/SVM/K_Fold_01_SMOTE",
    #     label_col="Result",   # bạn dùng "Result" chứ không phải "Dataset"
    #     random_state=42
    # )
    # SVM_experiment(
    #     output_file_name="indian_liver_patient_k_fold_02_SMOTE_experiment_result",
    #     train_path="../Data/data_apply_SMOTE/KFold_data/indian_liver_patient_train_k_fold_02_SMOTE.csv",
    #     test_path="../Data/KFold_data/indian_liver_patient_test_k_fold_02.csv",
    #     output_dir="../experiment_result/indian_liver_patient/SVM/K_Fold_02_SMOTE",
    #     label_col="Result",   # bạn dùng "Result" chứ không phải "Dataset"
    #     random_state=42
    # )
    # SVM_experiment(
    #     output_file_name="indian_liver_patient_k_fold_03_SMOTE_experiment_result",
    #     train_path="../Data/data_apply_SMOTE/KFold_data/indian_liver_patient_train_k_fold_03_SMOTE.csv",
    #     test_path="../Data/KFold_data/indian_liver_patient_test_k_fold_03.csv",
    #     output_dir="../experiment_result/indian_liver_patient/SVM/K_Fold_03_SMOTE",
    #     label_col="Result",   # bạn dùng "Result" chứ không phải "Dataset"
    #     random_state=42
    # )
    # SVM_experiment(
    #     output_file_name="indian_liver_patient_k_fold_04_SMOTE_experiment_result",
    #     train_path="../Data/data_apply_SMOTE/KFold_data/indian_liver_patient_train_k_fold_04_SMOTE.csv",
    #     test_path="../Data/KFold_data/indian_liver_patient_test_k_fold_04.csv",
    #     output_dir="../experiment_result/indian_liver_patient/SVM/K_Fold_04_SMOTE",
    #     label_col="Result",   # bạn dùng "Result" chứ không phải "Dataset"
    #     random_state=42
    # )
    # SVM_experiment(
    #     output_file_name="indian_liver_patient_k_fold_05_SMOTE_experiment_result",
    #     train_path="../Data/data_apply_SMOTE/KFold_data/indian_liver_patient_train_k_fold_05_SMOTE.csv",
    #     test_path="../Data/KFold_data/indian_liver_patient_test_k_fold_05.csv",
    #     output_dir="../experiment_result/indian_liver_patient/SVM/K_Fold_05_SMOTE",
    #     label_col="Result",   # bạn dùng "Result" chứ không phải "Dataset"
    #     random_state=42
    # )
    # SVM for liver cirhosis K_Fold = 5
    # SVM_experiment(
    #     output_file_name="liver_cirrhosis_k_fold_01_experiment_result",
    #     train_path="../Data/KFold_data/liver_cirrhosis_train_k_fold_01.csv",
    #     test_path="../Data/KFold_data/liver_cirrhosis_test_k_fold_01.csv",
    #     output_dir="../experiment_result/liver_cirrhosis/SVM/K_Fold_01",
    #     label_col="Stage",   
    #     random_state=42
    # )
    # SVM_experiment(
    #     output_file_name="liver_cirrhosis_k_fold_02_experiment_result",
    #     train_path="../Data/KFold_data/liver_cirrhosis_train_k_fold_02.csv",
    #     test_path="../Data/KFold_data/liver_cirrhosis_test_k_fold_02.csv",
    #     output_dir="../experiment_result/liver_cirrhosis/SVM/K_Fold_02",
    #     label_col="Stage",   
    #     random_state=42
    # )
    # SVM_experiment(
    #     output_file_name="liver_cirrhosis_k_fold_03_experiment_result",
    #     train_path="../Data/KFold_data/liver_cirrhosis_train_k_fold_03.csv",
    #     test_path="../Data/KFold_data/liver_cirrhosis_test_k_fold_03.csv",
    #     output_dir="../experiment_result/liver_cirrhosis/SVM/K_Fold_03",
    #     label_col="Stage",   
    #     random_state=42
    # )
    # SVM_experiment(
    #     output_file_name="liver_cirrhosis_k_fold_04_experiment_result",
    #     train_path="../Data/KFold_data/liver_cirrhosis_train_k_fold_04.csv",
    #     test_path="../Data/KFold_data/liver_cirrhosis_test_k_fold_04.csv",
    #     output_dir="../experiment_result/liver_cirrhosis/SVM/K_Fold_04",
    #     label_col="Stage",   
    #     random_state=42
    # )
    # SVM_experiment(
    #     output_file_name="liver_cirrhosis_k_fold_05_experiment_result",
    #     train_path="../Data/KFold_data/liver_cirrhosis_train_k_fold_05.csv",
    #     test_path="../Data/KFold_data/liver_cirrhosis_test_k_fold_05.csv",
    #     output_dir="../experiment_result/liver_cirrhosis/SVM/K_Fold_05",
    #     label_col="Stage",   
    #     random_state=42
    # )
    # SVM for indian liver disease K_Fold = 5 and SMOTE
    SVM_experiment(
        output_file_name="liver_cirrhosis_k_fold_01_SMOTE_experiment_result",
        train_path="../Data/data_apply_SMOTE/KFold_data/liver_cirrhosis_train_k_fold_01_SMOTE.csv",
        test_path="../Data/KFold_data/liver_cirrhosis_test_k_fold_01.csv",
        output_dir="../experiment_result/liver_cirrhosis/SVM/K_Fold_01_SMOTE",
        label_col="Stage",   # bạn dùng "Result" chứ không phải "Dataset"
        random_state=42
    )
    SVM_experiment(
        output_file_name="liver_cirrhosis_k_fold_02_SMOTE_experiment_result",
        train_path="../Data/data_apply_SMOTE/KFold_data/liver_cirrhosis_train_k_fold_02_SMOTE.csv",
        test_path="../Data/KFold_data/liver_cirrhosis_test_k_fold_02.csv",
        output_dir="../experiment_result/liver_cirrhosis/SVM/K_Fold_02_SMOTE",
        label_col="Stage",   # bạn dùng "Result" chứ không phải "Dataset"
        random_state=42
    )
    SVM_experiment(
        output_file_name="liver_cirrhosis_k_fold_03_SMOTE_experiment_result",
        train_path="../Data/data_apply_SMOTE/KFold_data/liver_cirrhosis_train_k_fold_03_SMOTE.csv",
        test_path="../Data/KFold_data/liver_cirrhosis_test_k_fold_03.csv",
        output_dir="../experiment_result/liver_cirrhosis/SVM/K_Fold_03_SMOTE",
        label_col="Stage",   # bạn dùng "Result" chứ không phải "Dataset"
        random_state=42
    )
    SVM_experiment(
        output_file_name="liver_cirrhosis_k_fold_04_SMOTE_experiment_result",
        train_path="../Data/data_apply_SMOTE/KFold_data/liver_cirrhosis_train_k_fold_04_SMOTE.csv",
        test_path="../Data/KFold_data/liver_cirrhosis_test_k_fold_04.csv",
        output_dir="../experiment_result/liver_cirrhosis/SVM/K_Fold_04_SMOTE",
        label_col="Stage",   # bạn dùng "Result" chứ không phải "Dataset"
        random_state=42
    )
    SVM_experiment(
        output_file_name="liver_cirrhosis_k_fold_05_SMOTE_experiment_result",
        train_path="../Data/data_apply_SMOTE/KFold_data/liver_cirrhosis_train_k_fold_05_SMOTE.csv",
        test_path="../Data/KFold_data/liver_cirrhosis_test_k_fold_05.csv",
        output_dir="../experiment_result/liver_cirrhosis/SVM/K_Fold_05_SMOTE",
        label_col="Stage",   # bạn dùng "Result" chứ không phải "Dataset"
        random_state=42
    )