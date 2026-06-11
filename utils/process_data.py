import sys
import os
import pandas as pd
import random
import copy
import pickle
import math
import numpy as np
import time
import torch

def preprocess_dataframe(df, target_col, cat_cols, id_cols=None):
    # Create a copy to avoid modifying the original dataframe in-place
    df = df.copy()
    
    # 0. Drop ID Columns
    if id_cols:
        # Only drop columns that actually exist in the dataframe to prevent errors
        cols_to_drop = [col for col in id_cols if col in df.columns]
        df = df.drop(columns=cols_to_drop)
        
    # 1. Drop rows with missing values (NaNs)
    # We do this after dropping IDs so we don't accidentally drop a good row 
    # just because it was missing an ID we didn't need anyway.
    df = df.dropna().reset_index(drop=True)
    
    # 2. Encode Non-Numeric Target Variable
    target_label_mapping = {}
    if target_col in cat_cols and not pd.api.types.is_numeric_dtype(df[target_col]):
        unique_classes = df[target_col].unique()
        target_label_mapping = {val: i for i, val in enumerate(unique_classes)}
        df[target_col] = df[target_col].map(target_label_mapping)
    
    # 3. Create Mapping & Rename
    features = [col for col in df.columns if col != target_col]
    
    col_name_mapping = {target_col: 'x0'}
    for i, col in enumerate(features, start=1):
        col_name_mapping[col] = f'x{i}'
        
    df = df.rename(columns=col_name_mapping)
    
    # 4. Order columns: target variable ('x0') first
    ordered_cols = ['x0'] + [f'x{i}' for i in range(1, len(features) + 1)]
    df = df[ordered_cols]
    
    # 5. One-Hot Encoding
    # Get the new names for the categorical columns, excluding the target
    mapped_cat_cols = [col_name_mapping[col] for col in cat_cols if col in col_name_mapping and col != target_col]
    
    if mapped_cat_cols:
        df = pd.get_dummies(df, columns=mapped_cat_cols, dtype=float)
        
    # 6. Min-Max Normalization
    cols_to_normalize = list(df.columns)
    
    # Exclude the target variable from normalization IF it was flagged as categorical
    if target_col in cat_cols and 'x0' in cols_to_normalize:
        cols_to_normalize.remove('x0')
        
    # Apply Min-Max scaling: (x - min) / (max - min)
    for col in cols_to_normalize:
        col_min = df[col].min()
        col_max = df[col].max()
        
        if col_max != col_min:
            df[col] = (df[col] - col_min) / (col_max - col_min)
        else:
            df[col] = 0.0 
            
    return df, col_name_mapping, target_label_mapping

def preprocess_dataframe_for_cases(df, target_col, cat_cols, id_cols=None):
    # Create a copy to avoid modifying the original dataframe in-place
    df = df.copy()
    
    # 0. Drop ID Columns
    if id_cols:
        # Only drop columns that actually exist in the dataframe to prevent errors
        cols_to_drop = [col for col in id_cols if col in df.columns]
        df = df.drop(columns=cols_to_drop)
        
    # 1. Drop rows with missing values (NaNs)
    # We do this after dropping IDs so we don't accidentally drop a good row 
    # just because it was missing an ID we didn't need anyway.
    df = df.dropna().reset_index(drop=True)
    
    # 2. Encode Non-Numeric Target Variable
    target_label_mapping = {}
    if target_col in cat_cols and not pd.api.types.is_numeric_dtype(df[target_col]):
        unique_classes = df[target_col].unique()
        target_label_mapping = {val: i for i, val in enumerate(unique_classes)}
        df[target_col] = df[target_col].map(target_label_mapping)
    
    # 3. Create Mapping & Rename
    features = [col for col in df.columns if col != target_col]
    
    col_name_mapping = {target_col: 'x0'}
    for i, col in enumerate(features, start=1):
        col_name_mapping[col] = f'x{i}'
        
    df = df.rename(columns=col_name_mapping)
    
    # 4. Order columns: target variable ('x0') first
    ordered_cols = ['x0'] + [f'x{i}' for i in range(1, len(features) + 1)]
    df = df[ordered_cols]
    
    # 5. One-Hot Encoding
    # Get the new names for the categorical columns, excluding the target
    mapped_cat_cols = [col_name_mapping[col] for col in cat_cols if col in col_name_mapping and col != target_col]
    
    if mapped_cat_cols:
        df = pd.get_dummies(df, columns=mapped_cat_cols, dtype=float)
        
    # 6. Min-Max Normalization
    cols_to_normalize = list(df.columns)
    
    # Exclude the target variable from normalization IF it was flagged as categorical
    if target_col in cat_cols and 'x0' in cols_to_normalize:
        cols_to_normalize.remove('x0')
        
    # Apply Min-Max scaling: (x - min) / (max - min)
    col_min_max = {}
    for col in cols_to_normalize:
        col_min = df[col].min()
        col_max = df[col].max()

        col_min_max[col] = [col_min, col_max]
        
        if col_max != col_min:
            df[col] = (df[col] - col_min) / (col_max - col_min)
        else:
            df[col] = 0.0 
            
    return df, col_name_mapping, target_label_mapping, col_min_max