# LinearModels_PCA_experiment.py (HOÀN CHỈNH - ĐẸP NHƯ CÁC MODEL KHÁC)
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

from models.Linear_Models import LinearModelClassifier


def LinearModels_PCA_experiment(
    input_path,
    output_dir,
    output_file_name="LinearModels_PCA_results",
    label_col="Result",
    n_splits=5,
    C_values=[0.01, 0.1, 1.0, 10.0],
    scalers=None,
    random_state=42
):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_path)
    X = df.drop(columns=[label_col])
    y = df[label_col].copy()

    classes = sorted(y.unique())
    n_classes = len(classes)
    is_multiclass = n_classes > 2
    if y.min() >= 1:
        y = y - y.min()
    class_names = [f"Stage {int(c + y.min())}" if "stage" in label_col.lower() else f"Class {int(c)}" for c in classes]

    # PCA
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X)
    total_var = pca.explained_variance_ratio_.sum()

    pca_vis_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    pca_vis_df['Label'] = df[label_col].values
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=pca_vis_df, x='PC1', y='PC2', hue='Label', palette='coolwarm', s=90, alpha=0.9)
    plt.title(f'Linear Models + PCA 2D (Variance: {total_var:.1%})', fontsize=14)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.legend(title=label_col)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "PCA_2D_Visualization.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # KFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    if scalers is None:
        scalers = {
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler(), 
            'RobustScaler': RobustScaler(), 
            'Normalizer': Normalizer(),
            'NoScaler': None
        }

    fold_results = []
    fold = 1

    for train_idx, test_idx in skf.split(X_pca, y):
        print(f"\nFold {fold}/{n_splits}")
        X_train, X_test = X_pca[train_idx], X_pca[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        for scaler_name, scaler in scalers.items():
            X_tr = scaler.fit_transform(X_train) if scaler else X_train
            X_te = scaler.transform(X_test) if scaler else X_test

            for C in C_values:
                model_name = f"Linear_C{C}"
                print(f"   [{scaler_name} + {model_name}] Training...")

                model_dir = os.path.join(output_dir, model_name)
                os.makedirs(model_dir, exist_ok=True)

                model = LinearModelClassifier(C=C, random_state=random_state)
                model.fit(X_tr, y_train)
                y_pred = model.predict(X_te)
                y_proba = model.predict_proba(X_te)

                acc = accuracy_score(y_test, y_pred)
                if is_multiclass:
                    pre = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1  = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    roc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
                else:
                    pre = precision_score(y_test, y_pred, zero_division=0)
                    rec = recall_score(y_test, y_pred, zero_division=0)
                    f1  = f1_score(y_test, y_pred, zero_division=0)
                    roc = roc_auc_score(y_test, y_proba[:, 1])

                fold_results.append({
                    'Fold': fold,
                    'Scaler': scaler_name,
                    'C': C,
                    'ACC': round(acc, 4),
                    'Precision': round(pre, 4),
                    'Recall': round(rec, 4),
                    'F1': round(f1, 4),
                    'ROC-AUC': round(roc, 4)
                })

                # ROC + CM (giống các model khác)
                plt.figure(figsize=(8, 6))
                if is_multiclass:
                    y_bin = label_binarize(y_test, classes=range(n_classes))
                    for i in range(n_classes):
                        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                        plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC={auc(fpr,tpr):.3f})')
                else:
                    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                    plt.plot(fpr, tpr, color='teal', lw=2, label=f'ROC (AUC={roc:.3f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.title(f'ROC - Fold {fold} | C={C} | {scaler_name}')
                plt.legend(loc="lower right")
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(model_dir, f"ROC_Fold{fold}_{scaler_name}.png"), dpi=300, bbox_inches='tight')
                plt.close()

                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(7, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
                plt.title(f'CM - Fold {fold} | C={C} | {scaler_name}')
                plt.savefig(os.path.join(model_dir, f"CM_Fold{fold}_{scaler_name}.png"), dpi=300, bbox_inches='tight')
                plt.close()

        fold += 1

    # Tổng hợp
    results_df = pd.DataFrame(fold_results)
    summary = results_df.groupby(['Scaler', 'C']).agg({
        'ACC': 'mean', 'Precision': 'mean', 'Recall': 'mean', 'F1': 'mean', 'ROC-AUC': 'mean'
    }).round(4).reset_index().sort_values('ROC-AUC', ascending=False)

    results_df.to_csv(os.path.join(output_dir, f"{output_file_name}_detailed.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, f"{output_file_name}_summary.csv"), index=False)

    print(f"\nLinear Models + PCA hoàn tất!")
    print(f"→ Best: C={summary.iloc[0]['C']}, Scaler={summary.iloc[0]['Scaler']}, ROC-AUC={summary.iloc[0]['ROC-AUC']}")

    return results_df, summary

if __name__ == "__main__":
  LinearModels_PCA_experiment(
      input_path="../Data/data_apply_SMOTE/indian_liver_patient_after_SMOTE.csv",
      output_dir="../experiment_result/indian_liver_patient/Linear_Models/PCA_5Fold_SMOTE",
      output_file_name="indian_liver_patient_SMOTE_PCA_LinearModels_results",
      label_col="Result",
      n_splits=5,
      random_state=42
  )
  LinearModels_PCA_experiment(
      input_path="../Data/data_apply_SMOTE/liver_cirrhosis_after_SMOTE.csv",
      output_dir="../experiment_result/liver_cirrhosis/Linear_Models/PCA_5Fold_SMOTE",
      output_file_name="liver_cirrhosis_SMOTE_PCA_LinearModels_results",
      label_col="Stage",
      n_splits=5,
      random_state=42
  )
