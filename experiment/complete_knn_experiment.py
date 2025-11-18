"""
=============================================================================
Complete workflow: Load → Preprocess → OOF K-Folds → Evaluate → Save
=============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, roc_auc_score,
    matthews_corrcoef, average_precision_score
)
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE

from KNN_module import KNNClassifier


# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    """Experiment configuration."""
    
    # Data
    DATA_PATH = "W://R-Stats/Liver_Disease_Dataset/ilpd_dataset.csv"
    ENCODING = 'latin1'
    TARGET_COL = "Result"
    
    # Preprocessing
    NAN_STRATEGY = "median"  # 'mean' or 'median'
    REMOVE_DUPLICATES = True
    USE_SMOTE = True
    
    # Scaler
    SCALER_TYPE = "robust"  # 'standard', 'robust', 'minmax'
    
    # Model
    N_NEIGHBORS = 5
    WEIGHTS = 'distance'
    METRIC = 'minkowski'
    P = 2
    
    # Experiment
    N_FOLDS = 5
    RANDOM_STATE = 42
    
    # Output
    OUTPUT_DIR = "W://ML_Outputs/"
    SAVE_PLOTS = True
    SAVE_CSV = True


# =============================================================================
# SCALER FACTORY
# =============================================================================
def get_scaler(name: str):
    """Get scaler by name."""
    scalers = {
        'standard': StandardScaler(),
        'robust': RobustScaler(),
        'minmax': MinMaxScaler()
    }
    if name not in scalers:
        raise ValueError(f"Invalid scaler: {name}. Choose from {list(scalers.keys())}")
    return scalers[name]


# =============================================================================
# DATA PREPROCESSING
# =============================================================================
def preprocess_data(
    filepath: str,
    target_col: str,
    nan_strategy: str = "median",
    remove_duplicates: bool = True,
    encoding: str = 'latin1'
) -> tuple:
    """
    Complete preprocessing pipeline.
    
    Returns:
        X, y, feature_names
    """
    print("="*80)
    print("DATA PREPROCESSING")
    print("="*80)
    
    # Load
    df = pd.read_csv(filepath, encoding=encoding)
    df.columns = df.columns.str.strip().str.replace('\xa0', '', regex=False)
    print(f"✔️ Loaded: {filepath}")
    print(f"   Shape: {df.shape}")
    
    # Auto-detect target if missing
    if target_col not in df.columns:
        possible = ["Result", "Dataset", "Class", "target", "Label"]
        for col in possible:
            if col in df.columns:
                target_col = col
                break
    
    # Normalize target to {0, 1}
    unique_vals = sorted(df[target_col].dropna().unique())
    if set(unique_vals) == {1, 2}:
        df[target_col] = df[target_col].map({1: 0, 2: 1})
    elif set(unique_vals) != {0, 1}:
        df[target_col] = (df[target_col] == max(unique_vals)).astype(int)
    
    # Drop NaN in target
    df = df.dropna(subset=[target_col])
    
    # Encode categorical
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if target_col in cat_cols:
        cat_cols.remove(target_col)
    
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    
    print(f"✔️ Encoded {len(cat_cols)} categorical columns")
    
    # Fill NaN
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)
    
    if num_cols:
        imputer = SimpleImputer(strategy=nan_strategy)
        df[num_cols] = imputer.fit_transform(df[num_cols])
    
    print(f"✔️ Filled NaN with {nan_strategy}")
    
    # Remove duplicates
    if remove_duplicates:
        before = len(df)
        df = df.drop_duplicates()
        print(f"✔️ Removed {before - len(df)} duplicates")
    
    # Separate X, y
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values.astype(int)
    feature_names = df.drop(columns=[target_col]).columns.tolist()
    
    print(f"✔️ Final shape: X={X.shape}, y={y.shape}")
    print("="*80)
    
    return X, y, feature_names


# =============================================================================
# K-FOLDS EXPERIMENT WITH OOF
# =============================================================================
class KNNExperiment:
    """K-Folds Cross-Validation with proper OOF evaluation."""
    
    def __init__(self, config: Config):
        self.config = config
        self.fold_results = []
        self.all_y_true = []
        self.all_y_pred = []
        self.all_y_proba = []
        self.classes_ = None
        self.train_time = 0
        self.test_time = 0
    
    def run(self, X: np.ndarray, y: np.ndarray):
        """Run K-Folds experiment with OOF."""
        print("\n" + "="*80)
        print(f"K-FOLDS EXPERIMENT (OOF) | K={self.config.N_NEIGHBORS}, Folds={self.config.N_FOLDS}")
        print("="*80)
        print("⚠️  Proper OOF: SMOTE & Scaler applied per fold on train only")
        
        self.classes_ = np.unique(y)
        skf = StratifiedKFold(
            n_splits=self.config.N_FOLDS,
            shuffle=True,
            random_state=self.config.RANDOM_STATE
        )
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            print(f"\n📊 Fold {fold}/{self.config.N_FOLDS}")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            print(f"   Original - Train: {len(X_train)}, Test: {len(X_test)}")
            
            # Apply SMOTE on train only
            if self.config.USE_SMOTE:
                smote = SMOTE(random_state=self.config.RANDOM_STATE)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                print(f"   After SMOTE - Train: {len(X_train)} (+{len(X_train)-len(X[train_idx])})")
            
            # Scale
            scaler = get_scaler(self.config.SCALER_TYPE)
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train
            t0 = time.time()
            knn = KNNClassifier(
                n_neighbors=self.config.N_NEIGHBORS,
                weights=self.config.WEIGHTS,
                metric=self.config.METRIC,
                p=self.config.P
            )
            knn.fit(X_train_scaled, y_train)
            t1 = time.time()
            
            # Predict
            y_pred = knn.predict(X_test_scaled)
            y_proba = knn.predict_proba(X_test_scaled)
            t2 = time.time()
            
            self.train_time += (t1 - t0)
            self.test_time += (t2 - t1)
            
            # Metrics
            metrics = self._calculate_metrics(y_test, y_pred, y_proba)
            metrics['fold'] = fold
            metrics['train_samples'] = len(X_train)
            metrics['test_samples'] = len(X_test)
            self.fold_results.append(metrics)
            
            # Store OOF
            self.all_y_true.extend(y_test)
            self.all_y_pred.extend(y_pred)
            self.all_y_proba.extend(y_proba)
            
            # Print
            print(f"   Acc: {metrics['accuracy']:.4f} | "
                  f"F1: {metrics['f1_score']:.4f} | "
                  f"AUC: {metrics['roc_auc']:.4f}")
        
        self._print_summary()
        return self
    
    def _calculate_metrics(self, y_true, y_pred, y_proba):
        """Calculate all metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'mcc': matthews_corrcoef(y_true, y_pred)
        }
        
        # ROC-AUC
        try:
            if len(self.classes_) == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                metrics['pr_auc'] = average_precision_score(y_true, y_proba[:, 1])
            else:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
                metrics['pr_auc'] = np.nan
        except:
            metrics['roc_auc'] = np.nan
            metrics['pr_auc'] = np.nan
        
        return metrics
    
    def _print_summary(self):
        """Print summary."""
        df = pd.DataFrame(self.fold_results)
        
        print("\n" + "="*80)
        print("SUMMARY RESULTS")
        print("="*80)
        print(f"\n{'Metric':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
        print("-"*80)
        
        for col in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc', 'mcc']:
            if col in df.columns:
                print(f"{col.upper():<15} "
                      f"{df[col].mean():<12.4f} "
                      f"{df[col].std():<12.4f} "
                      f"{df[col].min():<12.4f} "
                      f"{df[col].max():<12.4f}")
        
        print(f"\nTrain Time: {self.train_time:.2f}s")
        print(f"Test Time:  {self.test_time:.2f}s")
        print("="*80)
    
    def save_results(self, filename: str):
        """Save results to CSV."""
        df = pd.DataFrame(self.fold_results)
        
        # Add summary
        summary = {col: df[col].mean() for col in df.columns if col != 'fold'}
        summary['fold'] = 'MEAN'
        df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
        
        summary_std = {col: df[col][:-1].std() for col in df.columns if col != 'fold'}
        summary_std['fold'] = 'STD'
        df = pd.concat([df, pd.DataFrame([summary_std])], ignore_index=True)
        
        df.to_csv(filename, index=False)
        print(f"\n✔️ Results saved to: {filename}")
    
    def plot_confusion_matrix(self, filename: str = 'confusion_matrix.png'):
        """Plot confusion matrix."""
        cm = confusion_matrix(self.all_y_true, self.all_y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.classes_, yticklabels=self.classes_)
        plt.title(f'Confusion Matrix - KNN (K={self.config.N_NEIGHBORS})',
                 fontweight='bold')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✔️ Confusion matrix saved: {filename}")
    
    def plot_roc_curve(self, filename: str = 'roc_curve.png'):
        """Plot ROC curve."""
        y_true = np.array(self.all_y_true)
        y_proba = np.array(self.all_y_proba)
        
        plt.figure(figsize=(10, 8))
        
        if len(self.classes_) == 2:
            pos_label = self.classes_[1]
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1], pos_label=pos_label)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        else:
            y_true_bin = label_binarize(y_true, classes=self.classes_)
            for i, cls in enumerate(self.classes_):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, linewidth=2,
                        label=f'Class {cls} (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - KNN (K={self.config.N_NEIGHBORS})', fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✔️ ROC curve saved: {filename}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Main execution."""
    config = Config()
    
    print("="*80)
    print("KNN EXPERIMENT PIPELINE")
    print("="*80)
    print(f"Data: {config.DATA_PATH}")
    print(f"K={config.N_NEIGHBORS}, Folds={config.N_FOLDS}, SMOTE={config.USE_SMOTE}")
    
    # Preprocess
    X, y, feature_names = preprocess_data(
        filepath=config.DATA_PATH,
        target_col=config.TARGET_COL,
        nan_strategy=config.NAN_STRATEGY,
        remove_duplicates=config.REMOVE_DUPLICATES,
        encoding=config.ENCODING
    )
    
    # Run experiment
    experiment = KNNExperiment(config)
    experiment.run(X, y)
    
    # Save outputs
    if config.SAVE_CSV:
        experiment.save_results(config.OUTPUT_DIR + 'knn_experiment_results.csv')
    
    if config.SAVE_PLOTS:
        experiment.plot_confusion_matrix(config.OUTPUT_DIR + 'confusion_matrix.png')
        experiment.plot_roc_curve(config.OUTPUT_DIR + 'roc_curve.png')
    
    print("\n" + "="*80)
    print("✔️ EXPERIMENT COMPLETED")
    print("="*80)


if __name__ == "__main__":
    main()