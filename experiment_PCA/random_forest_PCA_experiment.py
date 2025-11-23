# RandomForest_PCA_experiment.py
import os
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.Random_Forest import RandomForestModel


def RandomForest_PCA_experiment(
    input_path,
    output_dir,
    output_file_name="RF_PCA_results",
    label_col="Result",
    n_splits=5,
    n_estimators_list=[100, 200],
    max_depth_list=[None, 10, 20],
    scalers=None,
    random_state=42
):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Đọc dữ liệu
    df = pd.read_csv(input_path)
    print(f"Đã tải: {input_path} | Shape: {df.shape}")

    X = df.drop(columns=[label_col])
    y = df[label_col].copy()

    # Phát hiện multiclass
    classes = sorted(y.unique())
    n_classes = len(classes)
    is_multiclass = n_classes > 2
    print(f"→ Nhiệm vụ: {'Multiclass' if is_multiclass else 'Binary'} ({n_classes} lớp: {classes})")

    # Chuẩn hóa nhãn về 0,1,2,...
    if y.min() >= 1:
        y = y - y.min()
    class_names = [f"Stage {int(c + y.min())}" if "stage" in label_col.lower() else f"Class {int(c)}" for c in classes]

    # 2. PCA 2D
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X)
    total_var = pca.explained_variance_ratio_.sum()
    print(f"→ PCA: Tổng variance explained = {total_var:.3%}")

    # Vẽ PCA Visualization
    pca_vis_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    pca_vis_df['Label'] = df[label_col].values
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=pca_vis_df, x='PC1', y='PC2', hue='Label', palette='tab10', s=90, alpha=0.9, edgecolor='k', linewidth=0.5)
    plt.title(f'Random Forest + PCA 2D Visualization\nAfter SMOTE | Variance Explained: {total_var:.1%}', fontsize=14)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.legend(title=label_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    pca_path = os.path.join(output_dir, "PCA_2D_Visualization.png")
    plt.savefig(pca_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"→ Đã lưu PCA visualization: {pca_path}")

    # 3. KFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    if scalers is None:
        scalers = {
            'NoScaler': None,
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'RobustScaler': RobustScaler(),
            'Normalizer': Normalizer()
        }

    fold_results = []
    fold = 1

    for train_idx, test_idx in skf.split(X_pca, y):
        print(f"\n=== Fold {fold}/{n_splits} ===")
        X_train, X_test = X_pca[train_idx], X_pca[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        for scaler_name, scaler in scalers.items():
            X_tr = scaler.fit_transform(X_train) if scaler else X_train.copy()
            X_te = scaler.transform(X_test)     if scaler else X_test.copy()

            for n_est in n_estimators_list:
                for depth in max_depth_list:
                    model_name = f"RF_n{n_est}_d{'None' if depth is None else depth}"
                    print(f"   [{scaler_name} + {model_name}] Training...")

                    model_dir = os.path.join(output_dir, model_name)
                    os.makedirs(model_dir, exist_ok=True)

                    model = RandomForestModel(n_estimators=n_est, max_depth=depth, random_state=random_state)
                    model.fit(X_tr, y_train)
                    y_pred = model.predict(X_te)
                    y_proba = model.predict_proba(X_te)

                    # Metrics
                    acc = accuracy_score(y_test, y_pred)
                    if is_multiclass:
                        pre = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                        f1  = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                        try:
                            roc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
                        except:
                            roc = np.nan
                    else:
                        pre = precision_score(y_test, y_pred, zero_division=0)
                        rec = recall_score(y_test, y_pred, zero_division=0)
                        f1  = f1_score(y_test, y_pred, zero_division=0)
                        roc = roc_auc_score(y_test, y_proba[:, 1])

                    fold_results.append({
                        'Fold': fold,
                        'Scaler': scaler_name,
                        'n_estimators': n_est,
                        'max_depth': 'None' if depth is None else depth,
                        'ACC': round(acc, 4),
                        'Precision': round(pre, 4),
                        'Recall': round(rec, 4),
                        'F1': round(f1, 4),
                        'ROC-AUC': round(roc, 4) if isinstance(roc, float) else 'N/A'
                    })

                    # === Vẽ ROC Curve ===
                    plt.figure(figsize=(8, 6))
                    if is_multiclass:
                        y_bin = label_binarize(y_test, classes=range(n_classes))
                        for i in range(n_classes):
                            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                            roc_auc_val = auc(fpr, tpr)
                            plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc_val:.3f})')
                        plt.title(f'ROC OvR - Fold {fold}\n{model_name} | {scaler_name}')
                    else:
                        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                        roc_auc_val = auc(fpr, tpr)
                        plt.plot(fpr, tpr, color='darkgreen', lw=2, label=f'ROC (AUC = {roc_auc_val:.3f})')
                    plt.plot([0, 1], [0, 1], 'k--', lw=2)
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.legend(loc="lower right")
                    plt.grid(True, alpha=0.3)
                    roc_path = os.path.join(model_dir, f"ROC_Fold{fold}_{scaler_name}.png")
                    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
                    plt.close()

                    # === Vẽ Confusion Matrix ===
                    cm = confusion_matrix(y_test, y_pred)
                    plt.figure(figsize=(7, 5))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                                xticklabels=class_names, yticklabels=class_names)
                    plt.title(f'Confusion Matrix - Fold {fold}\n{model_name} | {scaler_name}')
                    plt.ylabel('True Label')
                    plt.xlabel('Predicted Label')
                    cm_path = os.path.join(model_dir, f"CM_Fold{fold}_{scaler_name}.png")
                    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
                    plt.close()

        fold += 1

    # 4. Tổng hợp kết quả
    results_df = pd.DataFrame(fold_results)

    # Trung bình theo cấu hình
    summary = results_df.groupby(['Scaler', 'n_estimators', 'max_depth']).agg({
        'ACC': 'mean',
        'Precision': 'mean',
        'Recall': 'mean',
        'F1': 'mean',
        'ROC-AUC': lambda x: np.mean([v for v in x if isinstance(v, float)])
    }).round(4).reset_index().sort_values(by='ROC-AUC', ascending=False)

    # 5. Lưu 2 file CSV
    detailed_path = os.path.join(output_dir, f"{output_file_name}_detailed.csv")
    summary_path  = os.path.join(output_dir, f"{output_file_name}_summary.csv")

    results_df.to_csv(detailed_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"\nHOÀN TẤT Random Forest + PCA + {n_splits}-Fold CV")
    print(f"→ Thư mục: {output_dir}")
    print(f"→ File chi tiết: {detailed_path}")
    print(f"→ File tóm tắt: {summary_path}")
    print(f"→ BEST MODEL: {summary.iloc[0]['n_estimators']} cây, depth={summary.iloc[0]['max_depth']}, {summary.iloc[0]['Scaler']}")
    print(f"   → ROC-AUC trung bình = {summary.iloc[0]['ROC-AUC']}, ACC = {summary.iloc[0]['ACC']}")

    return results_df, summary

if __name__ == "__main__":
  RandomForest_PCA_experiment(
      input_path="../Data/data_apply_SMOTE/indian_liver_patient_after_SMOTE.csv",
      output_dir="../experiment_result/indian_liver_patient/Random_Forest/PCA_5Fold_SMOTE",
      output_file_name="indian_liver_patient_SMOTE_PCA_SVM_results",
      label_col="Result",
      n_splits=5,
      random_state=42
  )
  RandomForest_PCA_experiment(
      input_path="../Data/data_apply_SMOTE/liver_cirrhosis_after_SMOTE.csv",
      output_dir="../experiment_result/liver_cirrhosis/Random_Forest/PCA_5Fold_SMOTE",
      output_file_name="liver_cirrhosis_SMOTE_PCA_SVM_results",
      label_col="Stage",
      n_splits=5,
      random_state=42
  )