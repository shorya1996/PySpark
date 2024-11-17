# src/preprocess.py
import pandas as pd
import numpy as np
from logger import get_logger

logger = get_logger(__name__)

def load_data(file_path):
    """Load the dataset from the CSV file."""
    try:
        data = pd.read_csv(file_path)
        data.replace("none", np.NaN, inplace=True)
        logger.info(f"Data loaded from {file_path} with shape {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def handle_missing_values(df, threshold=0.5):
    """Remove columns with a high proportion of missing values."""
    drop_cols = df.columns[df.isnull().mean() > threshold]
    
    if len(drop_cols) > 0:
        logger.info(f"Dropping columns due to high missing values: {list(drop_cols)}")
    
    df.drop(columns=drop_cols, inplace=True)
    df.dropna(inplace=True)
    logger.info(f"After dropping missing values, data shape: {df.shape}")
    return df

def preprocess_data(data, target_column):
    """Prepare the dataset for model training."""
    y = data[target_column].apply(lambda x: 0 if x == 'good' else 1)
    X = data.drop(columns=[target_column])
    logger.info(f"Preprocessed data: {X.shape[1]} features")
    return X, y
