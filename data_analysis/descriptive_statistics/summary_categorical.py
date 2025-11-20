# descriptive_statistics/summary_categorical.py
import pandas as pd
from typing import List, Optional
from .confidence_interval import proportion_ci


def summarize_categorical(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    confidence: float = 0.95,
    wilson: bool = True
) -> pd.DataFrame:
    """
    Generate a comprehensive summary table for categorical variables.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    columns : list of str, optional
        Columns to summarize. If ``None``, all object/category columns are used.
    confidence : float
        Confidence level for proportion CIs.
    wilson : bool
        Use Wilson score interval (recommended) instead of normal approximation.

    Returns
    -------
    pd.DataFrame
        Long-format table with columns:
        - variable
        - level
        - frequency
        - percentage
        - ci_lower
        - ci_upper

    Notes
    -----
    Missing values are excluded from frequency counts but reported separately.
    """
    if columns is None:
        columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    records = []
    for col in columns:
        counts = df[col].value_counts(dropna=False)
        total = len(df)
        missing = counts.get(pd.NA, 0) + counts.get(np.nan, 0) + counts.get(None, 0)

        valid = df[col].dropna()
        n_valid = len(valid)

        for level, freq in counts.items():
            if pd.isna(level):
                records.append({
                    "variable": col,
                    "level": "Missing",
                    "frequency": freq,
                    "percentage": freq / total * 100,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan
                })
                continue

            perc = freq / n_valid * 100
            method = "wilson" if wilson else "normal"
            _, lower, upper = proportion_ci(freq, n_valid, confidence, method)
            records.append({
                "variable": col,
                "level": level,
                "frequency": freq,
                "percentage": perc,
                "ci_lower": lower * 100,
                "ci_upper": upper * 100
            })

    return pd.DataFrame(records)