# plots/distribution_plots/dist_plots.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
import pandas as pd
from typing import Optional, List, Tuple
import warnings
warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
palette = sns.color_palette("deep")

def plot_histogram(
    data: np.ndarray,
    col_name: str,
    bins: int = 30,
    kde: bool = True,
    figsize: Tuple[int, int] = (8, 5),
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot histogram with optional KDE overlay.

    Parameters
    ----------
    data : np.ndarray
        1D array of numeric values.
    col_name : str
        Column name for title and labeling.
    bins : int
        Number of histogram bins.
    kde : bool
        Overlay Kernel Density Estimate.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure (PNG/PDF, 300 DPI).

    Returns
    -------
    fig, ax

    References
    ----------
    .. [1] Silverman, B. W. (1986). Density Estimation for Statistics and Data Analysis.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(data, bins=bins, kde=kde, ax=ax, color=palette[0], alpha=0.7, stat="density")
    ax.set_title(f"Distribution of {col_name}", fontsize=14, pad=15)
    ax.set_xlabel(col_name)
    ax.set_ylabel("Density")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax


def plot_qq(
    data: np.ndarray,
    col_name: str,
    figsize: Tuple[int, int] = (6, 6),
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Q-Q plot against normal distribution.

    Parameters
    ----------
    data : np.ndarray
        Sample data.
    col_name : str
        Column name.
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    fig, ax

    References
    ----------
    .. [1] Wilk, M. B., & Gnanadesikan, R. (1968). Probability plotting methods for
           the analysis of data. Biometrika, 55(1), 1–17.
    """
    fig, ax = plt.subplots(figsize=figsize)
    stats.probplot(data, dist="norm", plot=ax)
    ax.get_lines()[1].set_color("red")
    ax.set_title(f"Q-Q Plot: {col_name} vs Normal", fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax


def plot_multiple_histograms(
    df,
    columns: List[str],
    target: Optional[str] = None,
    nrows: int = 2,
    ncols: int = 3,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Grid of histograms, optionally stratified by target.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list
        Numeric columns to plot.
    target : str, optional
        Binary/categorical column to hue.
    nrows, ncols : int
    figsize : tuple
    save_path : str

    Returns
    -------
    fig, axes
    """
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