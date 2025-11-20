# plots/relation_plots/rel_plots.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List

def plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue: Optional[str] = None,
    size: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Scatter plot with optional hue and size encoding.

    Parameters
    ----------
    df : pd.Data the DataFrame
    x_col, y_col : str
    hue, size : str, optional
    figsize : tuple
    save_path : str

    Returns
    -------
    fig, ax
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue, size=size, ax=ax, alpha=0.7, palette="viridis")

    ax.set_title(f"{y_col} vs {xocat_col}", fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax


def plot_correlation_heatmap(
    df: pd.DataFrame,
    method: str = 'pearson',
    annot: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Correlation matrix heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        Numeric columns only.
    method : {'pearson', 'spearman', 'kendall'}
    annot : bool
        Show correlation values.
    figsize : tuple
    save_path : str

    Returns
    -------
    fig, ax

    References
    ----------
    .. [1] Pearson, K. (1895). Contributions to the mathematical theory of evolution.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr(method=method)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=annot, cmap='coolwarm', center=0, ax=ax,
                square=True, cbar_kws={"shrink": .8}, fmt=".2f")
    ax.set_title(f"{method.capitalize()} Correlation Matrix", fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax


def plot_pairgrid(
    df: pd.DataFrame,
    vars: List[str],
    hue: Optional[str] = None,
    save_path: Optional[str] = None
) -> sns.PairGrid:
    """
    PairGrid with scatter (upper), KDE (diag), correlation (lower).

    Parameters
    ----------
    df : pd.DataFrame
    vars : list
        Numeric variables.
    hue : str
    save_path : str

    Returns
    -------
    g : sns.PairGrid
    """
    g = sns.PairGrid(df, vars=vars, hue=hue, diag_sharey=False)
    g.map_upper(sns.scatterplot, alpha=0.6)
    g.map_diag(sns.kdeplot, fill=True)
    g.map_lower(sns.kdeplot, cmap="Blues_d")
    g.add_legend()

    if save_path:
        g.savefig(save_path, dpi=300, bbox_inches='tight')
    return g