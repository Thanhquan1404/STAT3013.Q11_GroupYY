# data_preprocessing/transformation.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy.stats import yeojohnson

def scale_features(df, method='standard', columns=None):
    df_scaled = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("method must be 'standard', 'minmax', or 'robust'")
    
    df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
    return df_scaled, scaler


def log_transform(df, columns=None, add_eps=1e-6):
    df_trans = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if (df[col] < 0).any():
            df_trans[col] = np.log1p(df_trans[col] + df_trans[col].min().abs() + add_eps)
        else:
            df_trans[col] = np.log1p(df_trans[col])
    return df_trans


def yeojohnson_transform(df, columns=None):
    df_trans = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        transformed, _ = yeojohnson(df_trans[col])
        df_trans[col] = transformed
    return df_trans