# effect_sizes/cramer_v.py
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from typing import Union, Tuple


def cramer_v(
    contingency_table: Union[pd.DataFrame, np.ndarray],
    correction: bool = True
) -> Tuple[float, str]:
    """
    Compute Cramér's V for association between two categorical variables.

    Parameters
    ----------
    contingency_table : pd.DataFrame or np.ndarray
        2x2 or larger contingency table.
    correction : bool, default True
        Apply bias correction for small samples.

    Returns
    -------
    v : float
        Cramér's V (0 to 1).
    strength : str
        Interpretation: 'negligible', 'weak', 'moderate', 'strong', 'very strong'

    Notes
    -----
    - :math:`V = \\sqrt{\\frac{\\chi^2 / n}{\\min(r-1, c-1)}}`
    - Correction: :math:`V^* = \\sqrt{\\frac{(\\chi^2 / n) - \\frac{(r-1)(c-1)}{n-1}}{\\min(r-1, c-1)}}`

    References
    ----------
    .. [1] Cramér, H. (1946). *Mathematical Methods of Statistics*. Princeton University Press.
    .. [2] Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*.
    """
    table = pd.DataFrame(contingency_table)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return np.nan, "invalid table"

    chi2, _, _, _ = chi2_contingency(table, correction=False)
    n = table.sum().sum()
    r, c = table.shape
    phi2 = chi2 / n
    v = np.sqrt(phi2 / min(r - 1, c - 1))

    # Bias correction
    if correction and n > 1:
        phi2_corr = max(0, phi2 - ((r - 1) * (c - 1)) / (n - 1))
        v_corr = np.sqrt(phi2_corr / min(r - 1, c - 1))
        v = v_corr

    v = min(v, 1.0)  # Cap at 1.0

    # Interpretation (Cohen's guidelines for V)
    if v < 0.1:
        strength = "negligible"
    elif v < 0.3:
        strength = "weak"
    elif v < 0.5:
        strength = "moderate"
    else:
        strength = "strong" if v < 0.7 else "very strong"

    return round(v, 4), strength