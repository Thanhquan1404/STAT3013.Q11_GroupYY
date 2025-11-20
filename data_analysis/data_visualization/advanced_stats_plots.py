# plots/advanced_stats_plots/adv_plots.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from scipy import stats

def plot_boxen(
    df: pd.DataFrame,
    x: str,
    y: str,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Boxen plot (letter-value plot) – better for large datasets.

    Parameters
    ----------
    df : pd.DataFrame
    x, y : str
    figsize : tuple
    save_path : str

    Returns
    -------
    fig, ax

    References
    ----------
    .. [1] Hofmann, H., et al. (2017). Letter-value plots: Alternatives to boxplots.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxenplot(data=df, x=x, y=y, ax=ax, palette="Set3")
    ax.set_title(f"{y} by {x} (Letter-Value Plot)", fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax


def plot_violin_stratified(
    df: pd.DataFrame,
    target: str,
    features: list,
    ncols: int = 3,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Violin plots of features stratified by target.

    Parameters
    ----------
    df : pd.DataFrame
    target : str
        Binary/categorical target.
    features : list
        Numeric features.
    ncols : int
    figsize : tuple
    save_path : str

    Returns
    -------
    fig, axes
    """
    nrows = (len(features) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if nrows > 1 else [axes] if nrows == 1 else axes

    for i, feat in enumerate(features):
        sns.violinplot(data=df, x=target, y=feat, ax=axes[i], palette="muted", inner="quartile")
        axes[i].set_title(f"{feat} by {target}")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Distribution by Target (Violin Plots)", fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, axes


def plot_shapiro_residuals(
    residuals: np.ndarray,
    figsize: Tuple[int, int] = (10, 4),
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    Normality diagnostics: histogram + Q-Q plot of residuals.

    Parameters
    ----------
    residuals : np.ndarray
    figsize : tuple
    save_path : str

    Returns
    -------
    fig, (ax1, ax2)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    sns.histplot(residuals, kde=True, ax=ax1, color="skyblue")
    ax1.set_title("Residuals Distribution")
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title("Q-Q Plot of Residuals")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, (ax1, ax2)