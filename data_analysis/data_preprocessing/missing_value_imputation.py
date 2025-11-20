# data_preprocessing/missing_value_imputation.py
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer

def impute_missing_values(df, strategy='median', knn_neighbors=5):
    df_imputed = df.copy()
    
    numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
    categorical_cols = df_imputed.select_dtypes(include=['object', 'category']).columns
    
    if strategy in ['median', 'mean']:
        imputer = SimpleImputer(strategy=strategy)
        df_imputed[numeric_cols] = imputer.fit_transform(df_imputed[numeric_cols])
    elif strategy == 'knn':
        imputer = KNNImputer(n_neighbors=knn_neighbors)
        df_imputed[numeric_cols] = imputer.fit_transform(df_imputed[numeric_cols])
    
    if len(categorical_cols) > 0:
        mode_imputer = SimpleImputer(strategy='most_frequent')
        df_imputed[categorical_cols] = mode_imputer.fit_transform(df_imputed[categorical_cols])
    
    return df_imputed