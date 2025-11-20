# plots/categorical_plots/cat_plots.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional, List, Tuple

sns.set_style("whitegrid")

def plot_countplot(
    df: pd.DataFrame,
    col: str,
    hue: Optional[str] = None,
    order: Optional[List] = None,
    figsize: Tuple[int, int] = (8, 5),
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Enhanced count plot with percentage labels.

    Parameters
    ----------
    df : pd.DataFrame
    col : str
        Categorical column.
    hue : str, optional
        Grouping variable.
    order : list, optional
        Order of categories.
    figsize : tuple
    save_path : str

    Returns
    -------
    fig, ax
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.countplot(data=df, x=col, hue=hue, order=order, ax=ax, palette="Set2")

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


def plot_stacked_bar(
    df: pd.DataFrame,
    cat_col: str,
    target_col: str,
    normalize: bool = True,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Stacked bar chart showing proportion of target within categories.

    Parameters
    ----------
    df : pd.DataFrame
    cat_col : str
        Categorical predictor.
    target_col : str
        Binary target.
    normalize : bool
        Show proportions instead of counts.
    figsize : tuple
    save_path : str

    Returns
    -------
    fig, ax
    """
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