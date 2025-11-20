# diagnostic_tests/normality.py
import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Dict, Any, Union
import warnings
warnings.filterwarnings("ignore")


def shapiro_wilk_test(
    data: Union[pd.Series, np.ndarray],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Shapiro-Wilk test for normality.

    Parameters
    ----------
    data : pd.Series or np.ndarray
        Sample data.
    alpha : float
        Significance level.

    Returns
    -------
    result : dict
        W statistic, p-value, decision, and sample size.

    Notes
    -----
    - Recommended for n ≤ 5000.
    - H0: Data is normally distributed.

    References
    ----------
    .. [1] Shapiro, S. S., & Wilk, M. B. (1965). An analysis of variance test
           for normality (complete samples). Biometrika, 52(3/4), 591–611.
    """
    series = pd.Series(data).dropna()
    n = len(series)

    if n < 3:
        return {"test": "Shapiro-Wilk", "error": "n < 3", "n": n}
    if n > 5000:
        return {"test": "Shapiro-Wilk", "warning": "n > 5000, use D'Agostino", "n": n}

    w_stat, p_val = stats.shapiro(series)

    return {
        "test": "Shapiro-Wilk",
        "w_statistic": round(w_stat, 4),
        "p_value": p_val,
        "n": n,
        "normal": p_val > alpha,
        "alpha": alpha
    }


def dagostino_k_squared_test(
    data: Union[pd.Series, np.ndarray],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    D'Agostino's K² test for normality (skewness + kurtosis).

    Parameters
    ----------
    data : pd.Series or np.ndarray
    alpha : float

    Returns
    -------
    result : dict

    Notes
    -----
    - Better for larger samples (n > 20).
    - Combines tests for skewness and kurtosis.

    References
    ----------
    .. [1] D'Agostino, R. B. (1971). An omnibus test of normality for
           moderate and large size samples. Biometrika, 58(2), 341–348.
    """
    series = pd.Series(data).dropna()
    n = len(series)

    if n < 20:
        return {"test": "D'Agostino K²", "warning": "n < 20", "n": n}

    k2, p_val = stats.normaltest(series)

    return {
        "test": "D'Agostino K²",
        "k2_statistic": round(k2, 4),
        "p_value": p_val,
        "n": n,
        "normal": p_val > alpha,
        "alpha": alpha
    }


def anderson_darling_test(
    data: Union[pd.Series, np.ndarray],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Anderson-Darling test for normality.

    Parameters
    ----------
    data : pd.Series or np.ndarray
    alpha : float

    Returns
    -------
    result : dict
        Includes critical values for 1%, 5%, 10%.

    References
    ----------
    .. [1] Anderson, T. W., & Darling, D. A. (1954). A test of goodness of fit.
           Journal of the American Statistical Association, 49(268), 765–769.
    """
    series = pd.Series(data).dropna()
    n = len(series)

    if n < 8:
        return {"test": "Anderson-Darling", "error": "n < 8", "n": n}

    result = stats.anderson(series, dist='norm')
    significance_levels = [0.10, 0.05, 0.025, 0.01, 0.005]
    critical_values = dict(zip(significance_levels, result.critical_values))

    return {
        "test": "Anderson-Darling",
        "ad_statistic": round(result.statistic, 4),
        "critical_values": critical_values,
        "n": n,
        "normal_at_5pct": result.statistic < result.critical_values[2],
        "alpha": alpha
    }


def test_normality_all(
    df: pd.DataFrame,
    columns: List[str] = None,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Run all three normality tests on multiple numeric columns.

    Returns
    -------
    pd.DataFrame
        One row per column and test.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns

    results = []
    for col in columns:
        data = df[col]

        # Shapiro-Wilk
        sw = shapiro_wilk_test(data, alpha)
        sw.update({"variable": col})
        results.append(sw)

        # D'Agostino
        da = dagostino_k_squared_test(data, alpha)
        da.update({"variable": col})
        results.append(da)

        # Anderson-Darling
        ad = anderson_darling_test(data, alpha)
        ad.update({"variable": col})
        results.append(ad)

    return pd.DataFrame(results)