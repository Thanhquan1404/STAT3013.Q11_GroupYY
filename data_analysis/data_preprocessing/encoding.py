# data_preprocessing/encoding.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import category_encoders as ce

def label_encode_columns(df, columns=None):
    df_encoded = df.copy()
    le_dict = {}
    
    if columns is None:
        columns = [col for col in df.select_dtypes(include=['object', 'category']).columns 
                  if df[col].nunique() <= 2]
    
    for col in columns:
        if col not in df.columns:
            continue
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        le_dict[col] = le
    
    return df_encoded, le_dict


def one_hot_encode_columns(df, columns=None, drop_first=True):
    df_encoded = df.copy()
    
    if columns is None:
        columns = [col for col in df.select_dtypes(include=['object', 'category']).columns 
                  if df[col].nunique() > 2]
    
    if len(columns) > 0:
        df_encoded = pd.get_dummies(df_encoded, columns=columns, drop_first=drop_first)
    
    return df_encoded


def target_encode_column(df, target_col, col_to_encode, smoothing=1.0):
    df_encoded = df.copy()
    global_mean = df[target_col].mean()
    agg = df.groupby(col_to_encode)[target_col].agg(['mean', 'count'])
    smooth = (agg['mean'] * agg['count'] + global_mean * smoothing) / (agg['count'] + smoothing)
    df_encoded[col_to_encode + '_target_enc'] = df_encoded[col_to_encode].map(smooth)
    return df_encoded, smooth.to_dict()