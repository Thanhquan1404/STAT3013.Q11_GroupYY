# data_preprocessing/outlier_detection.py
import pandas as pd
import numpy as np
from scipy import stats

def detect_outliers_iqr(df, columns=None, factor=1.5):
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    outlier_masks = []
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        mask = (df[col] < lower) | (df[col] > upper)
        outlier_masks.append(mask)
    
    return pd.concat(outlier_masks, axis=1).any(axis=1)


def detect_outliers_zscore(df, columns=None, threshold=3):
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    z_scores = np.abs(stats.zscore(df[columns].dropna()))
    return (z_scores > threshold).any(axis=1)


def cap_outliers(df, columns=None, method='iqr', factor=1.5):
    df_capped = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - factor * IQR
            upper = Q3 + factor * IQR
        elif method == 'percentile':
            lower = df[col].quantile(0.05)
            upper = df[col].quantile(0.95)
        
        df_capped[col] = df_capped[col].clip(lower=lower, upper=upper)
    
    return df_capped