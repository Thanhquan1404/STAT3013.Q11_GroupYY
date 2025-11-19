import os
import sys
import time
import numpy as np
import pandas as pd
import importlib.util

# Import thư viện ML
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# ===========================================================================
# 1. SETUP ĐƯỜNG DẪN (Tự động tìm về thư mục gốc dự án)
# ===========================================================================
current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)       # Folder experiment
project_root = os.path.dirname(src_dir)       # Folder Root (liver-disease-ml-research)
sys.path.append(project_root) 

# IMPORT MODEL (File nằm trong src/models/svm_model.py)
try:
    from src.models.svm_model import SVMClassifier
except ImportError:
    # Fallback phòng hờ
    try:
        spec = importlib.util.spec_from_file_location("SVMClassifier", os.path.join(project_root, "src", "models", "svm_model.py"))
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            SVMClassifier = module.SVMClassifier
    except: pass

if 'SVMClassifier' not in locals() or SVMClassifier is None:
    print("❌ Lỗi Critical: Không tìm thấy src/models/svm_model.py"); sys.exit(1)

# ===========================================================================
# 2. CẤU HÌNH DỮ LIỆU (ĐÃ SỬA TÊN FILE CHO KHỚP GIT )
# ===========================================================================
DATASET_CONFIGS = [
    {
        "name": "Cirrhosis",
        # ⚠️ Đã sửa từ 'liver_cirrhosis.csv' thành 'cirrhosis.csv' để khớp file Git
        "input_path": os.path.join(project_root, "data", "raw", "cirrhosis.csv"),
        "output_path": os.path.join(project_root, "results", "metrics", "svm_cirrhosis_summary.csv")
    },
    {
        "name": "Indian Liver",
        "input_path": os.path.join(project_root, "data", "raw", "indian_liver_patient.csv"),
        "output_path": os.path.join(project_root, "results", "metrics", "svm_indian_liver_summary.csv")
    }
]

# ===========================================================================
# 3. HÀM CONFIGS & PREPROCESSING
# ===========================================================================
def get_configs():
    configs = []
    # Linear
    for c in [1, 10]: configs.append({'kernel': 'linear', 'C': c})
    # RBF
    for c in [10]:
        for g in [1, 0.5]: configs.append({'kernel': 'rbf', 'C': c, 'gamma': g})
    # Poly
    for c in [10]:
        for d in [2, 3]: configs.append({'kernel': 'poly', 'C': c, 'degree': d, 'gamma': 'scale'})
    return configs

def get_scaler(name):
    return {"StandardScaler": StandardScaler(), "MinMaxScaler": MinMaxScaler(), "RobustScaler": RobustScaler()}.get(name)

def process_data(df, scaler_name):
    possible_labels = ["Result", "Dataset", "Class", "selector", "target", "Stage", "status", "Diagnosis"]
    label_col = next((c for c in possible_labels if c in df.columns), None)
    if not label_col: return None

    df = df.dropna(subset=[label_col]).reset_index(drop=True)
    y = df[label_col]
    
    if len(np.unique(y)) > 2 or label_col == "Stage":
        task = "multiclass"; y = y.astype(int).values
    else:
        task = "binary"; y = (y == y.max()).astype(int).values

    X = df.drop(columns=[label_col, 'Status', 'N_Days', 'id', 'ID'], errors='ignore')
    
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].fillna("Missing").astype(str))
    
    num_cols = X.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        X[num_cols] = SimpleImputer(strategy="mean").fit_transform(X[num_cols])

    scaler = get_scaler(scaler_name)
    if scaler: X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    smote_stt = "OFF"
    try:
        if task == 'binary': 
            sm = SMOTE(random_state=42, k_neighbors=min(5, min(np.bincount(y_train))-1))
            X_train, y_train = sm.fit_resample(X_train, y_train)
            smote_stt = "ON"
    except: pass

    return X_train, y_train, X_test, y_test, task, smote_stt

# ===========================================================================
# 4. MAIN LOOP (HIỂN THỊ ĐẦY ĐỦ CV, TEST, F1)
# ===========================================================================
if __name__ == "__main__":
    SVM_CONFIGS = get_configs()
    SCALERS = ["StandardScaler", "MinMaxScaler"]

    for config in DATASET_CONFIGS:
        print(f"\n{'='*80}")
        print(f"📂 DATASET: {config['name']}")
        
        # Tạo folder results/metrics nếu chưa có 
        os.makedirs(os.path.dirname(config['output_path']), exist_ok=True)
        
        if not os.path.exists(config['output_path']):
            with open(config['output_path'], "w", encoding="utf-8") as f:
                f.write("Dataset,Task,Kernel,C,Gamma,Degree,CV_Acc,Test_Acc,Test_F1,Time_Train,Time_Test,Scaler\n")

        try:
            df_master = pd.read_csv(config['input_path'])
            print(f"✅ Đã load file: {os.path.basename(config['input_path'])}")
        except: 
            print(f"❌ Lỗi: Không tìm thấy file {config['input_path']}"); continue

        total_runs = len(SCALERS) * len(SVM_CONFIGS)
        current_run = 0

        for scaler_name in SCALERS:
            res = process_data(df_master, scaler_name)
            if not res: continue
            X_train, y_train, X_test, y_test, task, smote = res

            for cfg in SVM_CONFIGS:
                current_run += 1
                param_str = f"C={cfg['C']}"
                if 'gamma' in cfg: param_str += f", g={cfg['gamma']}"
                if 'degree' in cfg: param_str += f", d={cfg['degree']}"
                
                # Hiển thị tiến độ
                print(f"   👉 [{current_run}/{total_runs}] {scaler_name:<15} | {cfg['kernel']:<6} | {param_str} ... ", end="", flush=True)
                
                model = SVMClassifier(verbose=False, **cfg)
                
                # 1. TÍNH CV ACCURACY
                try:
                    cv_acc = np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1))
                except: cv_acc = 0.0
                
                # 2. TRAIN & TEST
                t0 = time.time(); model.fit(X_train, y_train); t_train = time.time() - t0
                t0 = time.time(); y_pred = model.predict(X_test); t_test = time.time() - t0
                
                # 3. TÍNH TEST ACC & F1 SCORE
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                test_acc = report['accuracy']
                f1 = report['weighted avg']['f1-score'] # F1 Weighted

                # In ra màn hình (Đúng yêu cầu của bạn)
                print(f"✅ CV: {cv_acc:.1%} | Test: {test_acc:.1%} | F1: {f1:.1%}")
                
                gamma_val = cfg.get('gamma', '')
                degree_val = cfg.get('degree', '')

                row = [config['name'], task, cfg['kernel'], cfg['C'], gamma_val, degree_val,
                       round(cv_acc,4), round(test_acc,4), round(f1,4), 
                       round(t_train,4), round(t_test,4), scaler_name]
                
                with open(config['output_path'], "a") as f:
                    f.write(",".join(map(str, row)) + "\n")

    print(f"\n🏁 HOÀN TẤT! Kết quả đã lưu vào results/metrics/")
