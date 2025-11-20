# effect_sizes/eta_squared.py
import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Union, Tuple


def eta_squared(
    groups: List[Union[pd.Series, np.ndarray]],
    correction: bool = True
) -> Tuple[float, str]:
    """
    Compute η² (eta-squared) for one-way ANOVA.

    Parameters
    ----------
    groups : list of pd.Series or np.ndarray
        k >= 2 independent groups.
    correction : bool, default True
        Use partial η² (recommended in medical research).

    Returns
    -------
    eta2 : float
        Effect size.
    magnitude : str
        Interpretation: 'small', 'medium', 'large'

    Notes
    -----
    - :math:`\\eta^2 = \\frac{SS_{between}}{SS_{total}}`
    - Partial η²: :math:`\\eta_p^2 = \\frac{SS_{between}}{SS_{between} + SS_{within}}`

    References
    ----------
    .. [1] Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*.
    .. [2] Lakens, D. (2013). Calculating and reporting effect sizes to facilitate
           cumulative science: a practical primer for t-tests and ANOVAs.
    """
    clean_groups = [pd.Series(g).dropna().values for g in groups]
    clean_groups = [g for g in clean_groups if len(g) > 0]

    if len(clean_groups) < 2:
        return np.nan, "insufficient groups"

    # Grand mean
    all_data = np.concatenate(clean_groups)
    grand_mean = all_data.mean()

    # Between-group sum of squares
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in clean_groups)

    # Total sum of squares
    ss_total = sum((x - grand_mean)**2 for g in clean_groups for x in g)

    if ss_total == 0:
        return 0.0, "no variance"

    eta2 = ss_between / ss_total

    # Partial eta-squared (recommended)
    if correction:
        ss_within = ss_total - ss_between
        partial_eta2 = ss_between / (ss_between + ss_within) if (ss_between + ss_within) > 0 else 0
        eta2 = partial_eta2

    # Interpretation (Cohen, 1988)
    if eta2 < 0.01:
        magnitude = "negligible"
    elif eta2 < 0.06:
        magnitude = "small"
    elif eta2 < 0.14:
        magnitude = "medium"
    else:
        magnitude = "large"

    return round(eta2, 4), magnitude