import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from typing import Optional, List, Tuple
import warnings
warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
palette = sns.color_palette("Set2")

# ===============================
# PLOTTING FUNCTIONS
# ===============================

def plot_boxplot(df: pd.DataFrame, col: str, hue: Optional[str]=None,
                 figsize: Tuple[int,int]=(8,5), save_path: Optional[str]=None) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=df, x=hue, y=col, ax=ax, palette=palette) if hue else sns.boxplot(data=df, y=col, ax=ax, palette=palette)
    ax.set_title(f"Boxplot of {col}" + (f" by {hue}" if hue else ""), fontsize=14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax

def plot_countplot(df: pd.DataFrame, col: str, hue: Optional[str] = None,
                   order: Optional[List] = None, figsize: Tuple[int,int]=(8,5),
                   save_path: Optional[str]=None) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    sns.countplot(data=df, x=col, hue=hue, order=order, ax=ax, palette=palette)
    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.text(p.get_x() + p.get_width()/2., height + 0.5,
                    f'{height}\n({height/total*100:.1f}%)',
                    ha="center", va="bottom", fontsize=9)
    ax.set_title(f"Distribution of {col}", fontsize=14)
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax

def plot_stacked_bar(df: pd.DataFrame, cat_col: str, target_col: str,
                     normalize: bool=True, figsize: Tuple[int,int]=(8,6),
                     save_path: Optional[str]=None) -> Tuple[plt.Figure, plt.Axes]:
    crosstab = pd.crosstab(df[cat_col], df[target_col], normalize='index' if normalize else None)
    fig, ax = plt.subplots(figsize=figsize)
    crosstab.plot(kind='bar', stacked=True, ax=ax, color=['#66c2a5', '#fc8d62'])
    ax.set_title(f"{target_col} by {cat_col}", fontsize=14)
    ax.set_xlabel(cat_col)
    ax.set_ylabel("Proportion" if normalize else "Count")
    ax.legend(title=target_col, loc='upper right')
    plt.xticks(rotation=0)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax

def plot_histogram(data: np.ndarray, col_name: str, bins: int=30, kde: bool=True,
                   figsize: Tuple[int,int]=(8,5), save_path: Optional[str]=None) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(data, bins=bins, kde=kde, ax=ax, color=palette[0], alpha=0.7, stat="density")
    ax.set_title(f"Distribution of {col_name}", fontsize=14, pad=15)
    ax.set_xlabel(col_name)
    ax.set_ylabel("Density")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax

def plot_qq(data: np.ndarray, col_name: str, figsize: Tuple[int,int]=(6,6),
            save_path: Optional[str]=None) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    stats.probplot(data, dist="norm", plot=ax)
    ax.get_lines()[1].set_color("red")
    ax.set_title(f"Q-Q Plot: {col_name} vs Normal", fontsize=14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax

def plot_multiple_histograms(df: pd.DataFrame, columns: List[str],
                             target: Optional[str]=None,
                             nrows: int=2, ncols: int=3,
                             figsize: Tuple[int,int]=(15,10),
                             save_path: Optional[str]=None) -> Tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    for i, col in enumerate(columns):
        if i >= len(axes):
            break
        ax = axes[i]
        if target is not None and target in df.columns:
            sns.histplot(data=df, x=col, hue=target, kde=True, ax=ax, alpha=0.6, stat="density")
        else:
            sns.histplot(data=df, x=col, kde=True, ax=ax, color=palette[0], alpha=0.7, stat="density")
        ax.set_title(col, fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel("")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Distribution of Numeric Features", fontsize=16, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, axes

def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, hue: Optional[str]=None,
                 size: Optional[str]=None, figsize: Tuple[int,int]=(8,6),
                 save_path: Optional[str]=None) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue, size=size, ax=ax, alpha=0.7, palette="viridis")
    ax.set_title(f"{y_col} vs {x_col}", fontsize=14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax

def plot_correlation_heatmap(df: pd.DataFrame, method: str='pearson', annot: bool=True,
                             figsize: Tuple[int,int]=(10,8), save_path: Optional[str]=None) -> Tuple[plt.Figure, plt.Axes]:
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr(method=method)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=annot, cmap='coolwarm', center=0, ax=ax, square=True, cbar_kws={"shrink": .8}, fmt=".2f")
    ax.set_title(f"{method.capitalize()} Correlation Matrix", fontsize=14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax

def plot_pairgrid(df: pd.DataFrame, vars: List[str], hue: Optional[str]=None,
                  save_path: Optional[str]=None):
    g = sns.PairGrid(df, vars=vars, hue=hue, diag_sharey=False)
    g.map_upper(sns.scatterplot, alpha=0.6)
    g.map_diag(sns.kdeplot, fill=True)
    g.map_lower(sns.kdeplot, cmap="Blues_d")
    g.add_legend()
    if save_path:
        g.savefig(save_path, dpi=300, bbox_inches='tight')
    return g

# ===============================
# PREPROCESSING
# ===============================

def get_scaler(name: str):
    if name.lower() == "standardscaler":
        return StandardScaler()
    elif name.lower() == "minmaxscaler":
        return MinMaxScaler()
    return None

def universal_preprocessing(path: str, scaler_name="StandardScaler",
                            apply_smote=True, random_state=42):
    print(f"\n[INFO] Loading dataset: {os.path.basename(path)}")
    df = pd.read_csv(path)
    possible_labels = ["Result", "Dataset", "Class", "selector", "target", "Stage", "status", "Diagnosis"]
    label_col = next((c for c in possible_labels if c in df.columns), None)
    if label_col is None:
        raise ValueError("Không tìm thấy cột nhãn!")

    y_raw = df[label_col].copy()
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

    if y_raw.isnull().any():
        df = df.dropna(subset=[label_col]).reset_index(drop=True)

    X = df.drop(columns=[label_col])
    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    num_cols = X.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        X[num_cols] = SimpleImputer(strategy="mean").fit_transform(X[num_cols])
    scaler = get_scaler(scaler_name)
    if scaler is not None:
        X = scaler.fit_transform(X)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)

    stratify = y if task == "binary" or n_classes > 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=stratify
    )
    if apply_smote and task == "binary":
        X_train, y_train = SMOTE(random_state=random_state).fit_resample(X_train, y_train)
        print(f"[INFO] SMOTE applied → X_train: {X_train.shape}")

    print(f"[INFO] Task: {task.upper()} | Classes: {n_classes} | Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    return X_train, y_train, X_test, y_test, task, label_col, df

# ===============================
# MAIN ANALYSIS FUNCTION
# ===============================

def analysis_process(dataset_path: str, output_dir: str,
                     scaler="StandardScaler", apply_smote=True):
    local_dt = datetime.now().astimezone()
    utc_dt = local_dt.astimezone(timezone.utc)
    print(f"\n[INFO] Start time (UTC): {utc_dt}")

    X_train, y_train, X_test, y_test, task, label_col, df_visual = universal_preprocessing(
        dataset_path, scaler_name=scaler, apply_smote=apply_smote
    )

    os.makedirs(output_dir, exist_ok=True)

    categorical_cols = df_visual.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = df_visual.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in numeric_cols:
        numeric_cols.remove(label_col)

    # 1. Countplots
    for col in categorical_cols:
        fig_path = os.path.join(output_dir, f"countplot_{col}.png")
        plot_countplot(df_visual, col, save_path=fig_path)

    # 2. Stacked bar (binary)
    if df_visual[label_col].nunique() == 2:
        for col in categorical_cols:
            if col == label_col: continue
            fig_path = os.path.join(output_dir, f"stackedbar_{col}.png")
            plot_stacked_bar(df_visual, col, label_col, save_path=fig_path)

    # 3. Multiple histograms
    fig_path = os.path.join(output_dir, "multiple_histograms.png")
    plot_multiple_histograms(df_visual, numeric_cols, target=label_col, save_path=fig_path)

    # 4. Individual histogram + QQ
    for col in numeric_cols:
        arr = df_visual[col].values
        hist_path = os.path.join(output_dir, f"hist_{col}.png")
        qq_path = os.path.join(output_dir, f"qq_{col}.png")
        plot_histogram(arr, col, save_path=hist_path)
        plot_qq(arr, col, save_path=qq_path)

    # 5. Scatter plot
    if len(numeric_cols) >= 2:
        x = numeric_cols[0]
        y = numeric_cols[1]
        fig_path = os.path.join(output_dir, f"scatter_{x}_vs_{y}.png")
        plot_scatter(df_visual, x, y, hue=label_col, save_path=fig_path)

    # 6. Correlation heatmap
    fig_path = os.path.join(output_dir, "correlation_heatmap.png")
    plot_correlation_heatmap(df_visual, save_path=fig_path)

    # 7. PairGrid (limit 4 numeric cols)
    select_for_pair = numeric_cols[:4]
    if len(select_for_pair) >= 2:
        fig_path = os.path.join(output_dir, "pairgrid.png")
        g = plot_pairgrid(df_visual, select_for_pair, hue=label_col)
        g.savefig(fig_path, dpi=300, bbox_inches='tight')
    # 8. Boxplots for numeric columns
    for col in numeric_cols:
        fig_path = os.path.join(output_dir, f"boxplot_{col}.png")
        plot_boxplot(df_visual, col, hue=label_col, save_path=fig_path)


    print("\n[INFO] Visualization complete.")
    print(f"[INFO] All plots saved to: {output_dir}")
    return df_visual

# ===============================
# EXAMPLE USAGE
# ===============================

if __name__ == "__main__":
    DATASET_PATH = "../../data/processed/indian_liver_patient_preprocessed.csv"
    OUTPUT_DIR = "eda_indian_liver/"

    # DATASET_PATH = "../../data/processed/liver_cirrhosis_preprocessed.csv"
    # OUTPUT_DIR = "eda_liver_cirrhosis/"

    analysis_process(DATASET_PATH, OUTPUT_DIR, apply_smote=False)
