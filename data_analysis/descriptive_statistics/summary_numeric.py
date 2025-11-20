# descriptive_statistics/summary_numeric.py
import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Optional
from .confidence_interval import mean_ci


def summarize_numeric(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    confidence: float = 0.95,
    iqr_factor: float = 1.5
) -> pd.DataFrame:
    """
    Produce a detailed numeric summary including central tendency,
    dispersion, normality test, and confidence intervals.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list of str, optional
        Numeric columns to summarize.
    confidence : float
        Confidence level for mean CI.
    iqr_factor : float
        Multiplier for outlier detection (default 1.5).

    Returns
    -------
    pd.DataFrame
        One row per variable with the following metrics:
        - n, n_missing, mean, std, median, min, max, q1, q3, iqr,
          skewness, kurtosis, shapiro_pvalue, mean_ci_lower, mean_ci_upper,
          n_outliers_iqr, pct_outliers_iqr
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    summary = []
    for col in columns:
        series = df[col]
        n = len(series)
        n_missing = series.isna().sum()
        clean = series.dropna()

        if len(clean) == 0:
            summary.append({ "variable": col, "n": n, "n_missing": n_missing })
            continue

        mean, ci_low, ci_up = mean_ci(clean, confidence)
        q1 = clean.quantile(0.25)
        q3 = clean.quantile(0.75)
        iqr_val = q3 - q1
        lower_fence = q1 - iqr_factor * iqr_val
        upper_fence = q3 + iqr_factor * iqr_val
        outliers = clean[(clean < lower_fence) | (clean > upper_fence)]
        n_outliers = len(outliers)

        shapiro_stat, shapiro_p = stats.shapiro(clean) if len(clean) <= 5000 else (np.nan, np.nan)

        row = {
            "variable": col,
            "n": n,
            "n_missing": n_missing,
            "mean": mean,
            "std": clean.std(),
            "median": clean.median(),
            "min": clean.min(),
            "max": clean.max(),
            "q1": q1,
            "q3": q3,
            "iqr": iqr_val,
            "skewness": stats.skew(clean),
            "kurtosis": stats.kurtosis(clean),
            "shapiro_pvalue": shapiro_p,
            "mean_ci_lower": ci_low,
            "mean_ci_upper": ci_up,
            "n_outliers_iqr": n_outliers,
            "pct_outliers_iqr": n_outliers / len(clean) * 100 if len(clean) > 0 else np.nan
        }
        summary.append(row)

    return pd.DataFrame(summary)