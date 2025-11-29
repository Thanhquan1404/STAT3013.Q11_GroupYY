# RandomForest_experiment.py
import os
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import warnings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from models.Random_Forest import RandomForestModel


def RandomForest_experiment(
    output_file_name,
    train_path,
    test_path,
    output_dir,
    label_col="Result",
    index_col=None,
    scalers=None,
    n_estimators_list=[100, 200],
    max_depth_list=[None, 10, 20],
    random_state=42
):
    os.makedirs(output_dir, exist_ok=True)

    train_df = pd.read_csv(train_path, index_col=index_col)
    test_df  = pd.read_csv(test_path, index_col=index_col)

    X_train = train_df.drop(columns=[label_col])
    y_train = train_df[label_col]
    X_test  = test_df.drop(columns=[label_col])
    y_test  = test_df[label_col]

    classes = sorted(np.unique(np.concatenate([y_train.unique(), y_test.unique()])))
    n_classes = len(classes)
    is_multiclass = n_classes > 2

    if y_train.min() >= 1:
        y_train = y_train - y_train.min()
        y_test = y_test - y_test.min()

    class_names = [f"Stage {int(c + y_train.min())}" if "stage" in label_col.lower() else f"Class {int(c)}" for c in classes]

    if scalers is None:
        scalers = {
            'NoScaler': None,
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'RobustScaler': RobustScaler(),
            'Normalizer': Normalizer()
        }

    results = []

    for scaler_name, scaler in scalers.items():
        X_tr = scaler.fit_transform(X_train) if scaler else X_train.values
        X_te = scaler.transform(X_test)     if scaler else X_test.values

        for n_est in n_estimators_list:
            for depth in max_depth_list:
                model_name = f"RF_{n_est}_{'None' if depth is None else depth}"
                print(f"[{scaler_name} + {model_name}] Training...")

                model_dir = os.path.join(output_dir, model_name)
                os.makedirs(model_dir, exist_ok=True)

                model = RandomForestModel(n_estimators=n_est, max_depth=depth, random_state=random_state)
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

                results.append({
                    'Scaler': scaler_name,
                    'n_estimators': n_est,
                    'max_depth': 'None' if depth is None else depth,
                    'ACC': round(acc, 4),
                    'Precision': round(pre, 4),
                    'Recall': round(rec, 4),
                    'F1': round(f1, 4),
                    'ROC-AUC': round(roc, 4)
                })

                # ROC
                plt.figure(figsize=(8, 6))
                if is_multiclass:
                    y_bin = label_binarize(y_test, classes=range(n_classes))
                    for i in range(n_classes):
                        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC={auc(fpr,tpr):.3f})')
                else:
                    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                    plt.plot(fpr, tpr, color='darkorange', label=f'ROC (AUC={roc:.3f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.title(f'{model_name} - {scaler_name}')
                plt.legend()
                plt.savefig(os.path.join(model_dir, f"ROC_{scaler_name}.png"), dpi=300, bbox_inches='tight')
                plt.close()

                # CM
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(7, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
                plt.title(f'CM - {model_name} - {scaler_name}')
                plt.savefig(os.path.join(model_dir, f"CM_{scaler_name}.png"), dpi=300, bbox_inches='tight')
                plt.close()

    results_df = pd.DataFrame(results).sort_values(by='ACC', ascending=False)
    results_df.to_csv(os.path.join(output_dir, f"{output_file_name}.csv"), index=False)
    print(f"Random Forest hoàn tất! Best ACC = {results_df.iloc[0]['ACC']}")
    return results_df

if __name__ == "__main__":
  # Random Forest experiment on KFold = 5
  # RandomForest_experiment(
  #   output_file_name="indian_liver_patient_k_fold_01_experiment_result",
  #   train_path="../Data/KFold_data/indian_liver_patient_train_k_fold_01.csv",
  #   test_path="../Data/KFold_data/indian_liver_patient_test_k_fold_01.csv",
  #   output_dir="../experiment_result/indian_liver_patient/Random_Forest/K_Fold_01",
  #   label_col="Result"
  # )
  # RandomForest_experiment(
  #   output_file_name="indian_liver_patient_k_fold_02_experiment_result",
  #   train_path="../Data/KFold_data/indian_liver_patient_train_k_fold_02.csv",
  #   test_path="../Data/KFold_data/indian_liver_patient_test_k_fold_02.csv",
  #   output_dir="../experiment_result/indian_liver_patient/Random_Forest/K_Fold_02",
  #   label_col="Result"
  # )
  # RandomForest_experiment(
  #   output_file_name="indian_liver_patient_k_fold_03_experiment_result",
  #   train_path="../Data/KFold_data/indian_liver_patient_train_k_fold_03.csv",
  #   test_path="../Data/KFold_data/indian_liver_patient_test_k_fold_03.csv",
  #   output_dir="../experiment_result/indian_liver_patient/Random_Forest/K_Fold_03",
  #   label_col="Result"
  # )
  # RandomForest_experiment(
  #   output_file_name="indian_liver_patient_k_fold_04_experiment_result",
  #   train_path="../Data/KFold_data/indian_liver_patient_train_k_fold_04.csv",
  #   test_path="../Data/KFold_data/indian_liver_patient_test_k_fold_04.csv",
  #   output_dir="../experiment_result/indian_liver_patient/Random_Forest/K_Fold_04",
  #   label_col="Result"
  # )
  # RandomForest_experiment(
  #   output_file_name="indian_liver_patient_k_fold_05_experiment_result",
  #   train_path="../Data/KFold_data/indian_liver_patient_train_k_fold_05.csv",
  #   test_path="../Data/KFold_data/indian_liver_patient_test_k_fold_05.csv",
  #   output_dir="../experiment_result/indian_liver_patient/Random_Forest/K_Fold_05",
  #   label_col="Result"
  # )
  # Random Forest for indian liver disease K_Fold = 5 and SMOTE
  # RandomForest_experiment(
  #     output_file_name="indian_liver_patient_k_fold_01_SMOTE_experiment_result",
  #     train_path="../Data/data_apply_SMOTE/KFold_data/indian_liver_patient_train_k_fold_01_SMOTE.csv",
  #     test_path="../Data/KFold_data/indian_liver_patient_test_k_fold_01.csv",
  #     output_dir="../experiment_result/indian_liver_patient/Random_Forest/K_Fold_01_SMOTE",
  #     label_col="Result",   # bạn dùng "Result" chứ không phải "Dataset"
  # )
  # RandomForest_experiment(
  #     output_file_name="indian_liver_patient_k_fold_02_SMOTE_experiment_result",
  #     train_path="../Data/data_apply_SMOTE/KFold_data/indian_liver_patient_train_k_fold_02_SMOTE.csv",
  #     test_path="../Data/KFold_data/indian_liver_patient_test_k_fold_02.csv",
  #     output_dir="../experiment_result/indian_liver_patient/Random_Forest/K_Fold_02_SMOTE",
  #     label_col="Result",   # bạn dùng "Result" chứ không phải "Dataset"
  # )
  # RandomForest_experiment(
  #     output_file_name="indian_liver_patient_k_fold_03_SMOTE_experiment_result",
  #     train_path="../Data/data_apply_SMOTE/KFold_data/indian_liver_patient_train_k_fold_03_SMOTE.csv",
  #     test_path="../Data/KFold_data/indian_liver_patient_test_k_fold_03.csv",
  #     output_dir="../experiment_result/indian_liver_patient/Random_Forest/K_Fold_03_SMOTE",
  #     label_col="Result",   # bạn dùng "Result" chứ không phải "Dataset"
  # )
  # RandomForest_experiment(
  #     output_file_name="indian_liver_patient_k_fold_04_SMOTE_experiment_result",
  #     train_path="../Data/data_apply_SMOTE/KFold_data/indian_liver_patient_train_k_fold_04_SMOTE.csv",
  #     test_path="../Data/KFold_data/indian_liver_patient_test_k_fold_04.csv",
  #     output_dir="../experiment_result/indian_liver_patient/Random_Forest/K_Fold_04_SMOTE",
  #     label_col="Result",   # bạn dùng "Result" chứ không phải "Dataset"
  # )
  # RandomForest_experiment(
  #     output_file_name="indian_liver_patient_k_fold_05_SMOTE_experiment_result",
  #     train_path="../Data/data_apply_SMOTE/KFold_data/indian_liver_patient_train_k_fold_05_SMOTE.csv",
  #     test_path="../Data/KFold_data/indian_liver_patient_test_k_fold_05.csv",
  #     output_dir="../experiment_result/indian_liver_patient/Random_Forest/K_Fold_05_SMOTE",
  #     label_col="Result",   # bạn dùng "Result" chứ không phải "Dataset"
  # )
  # SVM for liver cirhosis K_Fold = 5
  RandomForest_experiment(
      output_file_name="liver_cirrhosis_k_fold_01_experiment_result",
      train_path="../Data/KFold_data/liver_cirrhosis_train_k_fold_01.csv",
      test_path="../Data/KFold_data/liver_cirrhosis_test_k_fold_01.csv",
      output_dir="../experiment_result/liver_cirrhosis/Random_Forest/K_Fold_01",
      label_col="Stage",   
  )
  RandomForest_experiment(
      output_file_name="liver_cirrhosis_k_fold_02_experiment_result",
      train_path="../Data/KFold_data/liver_cirrhosis_train_k_fold_02.csv",
      test_path="../Data/KFold_data/liver_cirrhosis_test_k_fold_02.csv",
      output_dir="../experiment_result/liver_cirrhosis/Random_Forest/K_Fold_02",
      label_col="Stage",   
  )
  RandomForest_experiment(
      output_file_name="liver_cirrhosis_k_fold_03_experiment_result",
      train_path="../Data/KFold_data/liver_cirrhosis_train_k_fold_03.csv",
      test_path="../Data/KFold_data/liver_cirrhosis_test_k_fold_03.csv",
      output_dir="../experiment_result/liver_cirrhosis/Random_Forest/K_Fold_03",
      label_col="Stage",   
  )
  RandomForest_experiment(
      output_file_name="liver_cirrhosis_k_fold_04_experiment_result",
      train_path="../Data/KFold_data/liver_cirrhosis_train_k_fold_04.csv",
      test_path="../Data/KFold_data/liver_cirrhosis_test_k_fold_04.csv",
      output_dir="../experiment_result/liver_cirrhosis/Random_Forest/K_Fold_04",
      label_col="Stage",   
  )
  RandomForest_experiment(
      output_file_name="liver_cirrhosis_k_fold_05_experiment_result",
      train_path="../Data/KFold_data/liver_cirrhosis_train_k_fold_05.csv",
      test_path="../Data/KFold_data/liver_cirrhosis_test_k_fold_05.csv",
      output_dir="../experiment_result/liver_cirrhosis/Random_Forest/K_Fold_05",
      label_col="Stage",   
  )
  # SVM for indian liver disease K_Fold = 5 and SMOTE
  RandomForest_experiment(
      output_file_name="liver_cirrhosis_k_fold_01_SMOTE_experiment_result",
      train_path="../Data/data_apply_SMOTE/KFold_data/liver_cirrhosis_train_k_fold_01_SMOTE.csv",
      test_path="../Data/KFold_data/liver_cirrhosis_test_k_fold_01.csv",
      output_dir="../experiment_result/liver_cirrhosis/Random_Forest/K_Fold_01_SMOTE",
      label_col="Stage",   # bạn dùng "Result" chứ không phải "Dataset"
  )
  RandomForest_experiment(
      output_file_name="liver_cirrhosis_k_fold_02_SMOTE_experiment_result",
      train_path="../Data/data_apply_SMOTE/KFold_data/liver_cirrhosis_train_k_fold_02_SMOTE.csv",
      test_path="../Data/KFold_data/liver_cirrhosis_test_k_fold_02.csv",
      output_dir="../experiment_result/liver_cirrhosis/Random_Forest/K_Fold_02_SMOTE",
      label_col="Stage",   # bạn dùng "Result" chứ không phải "Dataset"
  )
  RandomForest_experiment(
      output_file_name="liver_cirrhosis_k_fold_03_SMOTE_experiment_result",
      train_path="../Data/data_apply_SMOTE/KFold_data/liver_cirrhosis_train_k_fold_03_SMOTE.csv",
      test_path="../Data/KFold_data/liver_cirrhosis_test_k_fold_03.csv",
      output_dir="../experiment_result/liver_cirrhosis/Random_Forest/K_Fold_03_SMOTE",
      label_col="Stage",   # bạn dùng "Result" chứ không phải "Dataset"
  )
  RandomForest_experiment(
      output_file_name="liver_cirrhosis_k_fold_04_SMOTE_experiment_result",
      train_path="../Data/data_apply_SMOTE/KFold_data/liver_cirrhosis_train_k_fold_04_SMOTE.csv",
      test_path="../Data/KFold_data/liver_cirrhosis_test_k_fold_04.csv",
      output_dir="../experiment_result/liver_cirrhosis/Random_Forest/K_Fold_04_SMOTE",
      label_col="Stage",   # bạn dùng "Result" chứ không phải "Dataset"
  )
  RandomForest_experiment(
      output_file_name="liver_cirrhosis_k_fold_05_SMOTE_experiment_result",
      train_path="../Data/data_apply_SMOTE/KFold_data/liver_cirrhosis_train_k_fold_05_SMOTE.csv",
      test_path="../Data/KFold_data/liver_cirrhosis_test_k_fold_05.csv",
      output_dir="../experiment_result/liver_cirrhosis/Random_Forest/K_Fold_05_SMOTE",
      label_col="Stage",   # bạn dùng "Result" chứ không phải "Dataset"
  )

  