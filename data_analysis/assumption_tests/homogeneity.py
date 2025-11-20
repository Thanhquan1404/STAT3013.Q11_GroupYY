# diagnostic_tests/homogeneity.py
import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Dict, Any, Union


def levene_test(
    groups: List[Union[pd.Series, np.ndarray]],
    center: str = "median",
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Levene's test for homogeneity of variances.

    Parameters
    ----------
    groups : list
        k >= 2 groups.
    center : {'median', 'mean', 'trimmed'}
        Robust to non-normality when using 'median'.
    alpha : float

    Returns
    -------
    result : dict

    Notes
    -----
    - H0: All groups have equal variance.
    - Use 'median' for non-normal data.

    References
    ----------
    .. [1] Levene, H. (1960). Robust tests for equality of variances.
           In Contributions to Probability and Statistics.
    """
    clean_groups = [pd.Series(g).dropna().values for g in groups]
    clean_groups = [g for g in clean_groups if len(g) > 1]

    if len(clean_groups) < 2:
        return {"test": "Levene", "error": "Need >= 2 groups with n > 1"}

    w_stat, p_val = stats.levene(*clean_groups, center=center)

    return {
        "test": "Levene",
        "w_statistic": round(w_stat, 4),
        "p_value": p_val,
        "center": center,
        "n_groups": len(clean_groups),
        "equal_variance": p_val > alpha,
        "alpha": alpha
    }


def bartlett_test(
    groups: List[Union[pd.Series, np.ndarray]],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Bartlett's test for homogeneity of variances (assumes normality).

    References
    ----------
    .. [1] Bartlett, M. S. (1937). Properties of sufficiency and statistical tests.
           Proceedings of the Royal Society of London.
    """
    clean_groups = [pd.Series(g).dropna().values for g in groups]
    clean_groups = [g for g in clean_groups if len(g) > 1]

    if len(clean_groups) < 2:
        return {"test": "Bartlett", "error": "Need >= 2 groups"}

    t_stat, p_val = stats.bartlett(*clean_groups)

    return {
        "test": "Bartlett",
        "t_statistic": round(t_stat, 4),
        "p_value": p_val,
        "n_groups": len(clean_groups),
        "equal_variance": p_val > alpha,
        "alpha": alpha
    }


def test_homogeneity_all(
    df: pd.DataFrame,
    group_col: str,
    value_cols: List[str] = None,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Test homogeneity across multiple numeric variables stratified by group.

    Returns
    -------
    pd.DataFrame
    """
    if value_cols is None:
        value_cols = df.select_dtypes(include=[np.number]).columns.drop(group_col, errors='ignore')

    results = []
    for col in value_cols:
        groups = [group[col].dropna() for name, group in df.groupby(group_col)]
        if len(groups) < 2:
            continue

        lev = levene_test(groups, center="median", alpha=alpha)
        lev["variable"] = col
        results.append(lev)

        bart = bartlett_test(groups, alpha=alpha)
        bart["variable"] = col
        results.append(bart)

    return pd.DataFrame(results)