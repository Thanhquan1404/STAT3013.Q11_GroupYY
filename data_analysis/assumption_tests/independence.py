# diagnostic_tests/independence.py
import pandas as pd
from scipy.stats import chi2_contingency
from typing import Dict, Any


def chi2_independence_test(
    contingency_table: pd.DataFrame,
    alpha: float = 0.05,
    correction: bool = True
) -> Dict[str, Any]:
    """
    Chi-squared test of independence.

    Parameters
    ----------
    contingency_table : pd.DataFrame
        Crosstab (rows = var1, cols = var2).
    alpha : float
    correction : bool
        Yates' continuity correction for 2x2 tables.

    Returns
    -------
    result : dict

    References
    ----------
    .. [1] Pearson, K. (1900). On the criterion that a given system of deviations...
    """
    chi2, p_val, dof, expected = chi2_contingency(contingency_table, correction=correction)

    return {
        "test": "Chi-squared Independence",
        "chi2_statistic": round(chi2, 4),
        "p_value": p_val,
        "df": dof,
        "expected_frequencies": expected.tolist(),
        "independent": p_val > alpha,
        "alpha": alpha,
        "correction": correction
    }


def fisher_exact_test(
    contingency_table: pd.DataFrame,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Fisher's exact test for 2x2 tables.

    References
    ----------
    .. [1] Fisher, R. A. (1922). On the interpretation of χ² from contingency tables.
    """
    if contingency_table.shape != (2, 2):
        return {"test": "Fisher Exact", "error": "Must be 2x2 table"}

    oddsratio, p_val = stats.fisher_exact(contingency_table)

    return {
        "test": "Fisher Exact",
        "odds_ratio": round(oddsratio, 4),
        "p_value": p_val,
        "independent": p_val > alpha,
        "alpha": alpha
    }