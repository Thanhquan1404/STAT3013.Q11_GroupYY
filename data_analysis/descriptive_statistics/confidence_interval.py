# descriptive_statistics/confidence_interval.py
import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple, Union


def mean_ci(
    data: Union[pd.Series, np.ndarray],
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute the confidence interval for the population mean
    using the t-distribution (appropriate for small samples).

    Parameters
    ----------
    data : pd.Series or np.ndarray
        Numeric sample data (non-missing values are used automatically).
    confidence : float, default 0.95
        Desired confidence level (0 < confidence < 1).

    Returns
    -------
    mean : float
        Sample mean.
    lower : float
        Lower bound of the CI.
    upper : float
        Upper bound of the CI.

    Notes
    -----
    The t-distribution is used because the population standard deviation
    is unknown. For large samples (n > 30) the result approximates the
    normal-distribution CI.

    References
    ----------
    .. [1] Student, "The probable error of a mean", Biometrika, 1908.
    """
    arr = pd.Series(data).dropna().values
    n = len(arr)
    if n == 0:
        return np.nan, np.nan, np.nan

    mean = arr.mean()
    se = stats.sem(arr)                     # standard error
    alpha = 1 - confidence
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    margin = t_crit * se
    return mean, mean - margin, mean + margin


def proportion_ci(
    successes: int,
    n: int,
    confidence: float = 0.95,
    method: str = "normal"
) -> Tuple[float, float, float]:
    """
    Confidence interval for a binomial proportion.

    Parameters
    ----------
    successes : int
        Number of successes.
    n : int
        Total number of trials.
    confidence : float
        Confidence level.
    method : {'normal', 'wilson'}, default 'normal'
        Approximation method. 'wilson' is recommended for small n or
        extreme proportions.

    Returns
    -------
    p_hat, lower, upper

    References
    ----------
    .. [1] Wilson, E. B. (1927). Probable inference, the law of succession,
           and statistical inference. J. Am. Stat. Assoc., 22, 209–212.
    """
    if n == 0:
        return np.nan, np.nan, np.nan
    p_hat = successes / n
    alpha = 1 - confidence

    if method == "normal":
        se = np.sqrt(p_hat * (1 - p_hat) / n)
        z = stats.norm.ppf(1 - alpha / 2)
        margin = z * se
    elif method == "wilson":
        z = stats.norm.ppf(1 - alpha / 2)
        denominator = 1 + z**2 / n
        centre = (p_hat + z**2 / (2 * n)) / denominator
        margin = (z / denominator) * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
        p_hat, lower, upper = centre, centre - margin, centre + margin
    else:
        raise ValueError("method must be 'normal' or 'wilson'")

    return p_hat, lower, upper