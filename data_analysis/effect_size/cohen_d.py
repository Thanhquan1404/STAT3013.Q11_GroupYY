# effect_sizes/cohen_d.py
import pandas as pd
import numpy as np
from typing import Union, Tuple, Literal


def cohen_d(
    group1: Union[pd.Series, np.ndarray],
    group2: Union[pd.Series, np.ndarray],
    pooled: bool = True,
    correction: bool = True
) -> Tuple[float, str]:
    """
    Compute Cohen's d effect size for two independent groups.

    Parameters
    ----------
    group1, group2 : pd.Series or np.ndarray
        Two independent samples.
    pooled : bool, default True
        Use pooled standard deviation (standard Cohen's d).
        If False, uses average standard deviation (for unequal variances).
    correction : bool, default True
        Apply Hedges' correction for small sample bias.

    Returns
    -------
    d : float
        Effect size.
    magnitude : str
        Interpretation: 'negligible', 'small', 'medium', 'large'

    Notes
    -----
    - Pooled SD: :math:`s_p = \\sqrt{\\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}}`
    - Hedges' g correction: :math:`g = d \\times (1 - \\frac{3}{4(n_1 + n_2) - 9})`

    References
    ----------
    .. [1] Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Routledge.
    .. [2] Hedges, L. V. (1981). Distribution theory for Glass's estimator of effect size and related estimators.
           Journal of Educational Statistics, 6(2), 107–128.
    """
    g1 = pd.Series(group1).dropna().values
    g2 = pd.Series(group2).dropna().values

    if len(g1) < 2 or len(g2) < 2:
        return np.nan, "insufficient data"

    n1, n2 = len(g1), len(g2)
    mean1, mean2 = g1.mean(), g2.mean()
    diff = mean1 - mean2

    if pooled:
        # Pooled standard deviation
        var1, var2 = g1.var(ddof=1), g2.var(ddof=1)
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        s_pooled = np.sqrt(pooled_var)
        d = diff / s_pooled
    else:
        # Average SD (for unequal variance)
        s_avg = (g1.std(ddof=1) + g2.std(ddof=1)) / 2
        d = diff / s_avg

    # Hedges' correction for small samples
    if correction:
        df = n1 + n2 - 2
        correction_factor = 1 - 3 / (4 * df - 1)
        d *= correction_factor

    # Interpretation (Cohen's conventions)
    abs_d = abs(d)
    if abs_d < 0.2:
        magnitude = "negligible"
    elif abs_d < 0.5:
        magnitude = "small"
    elif abs_d < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"

    return round(d, 4), magnitude