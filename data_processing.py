# src/preprocess.py
import pandas as pd
import numpy as np

def load_data(file_path):
    """Load the dataset from the CSV file."""
    data = pd.read_csv(file_path)
    data.replace("none", np.NaN, inplace=True)
    return data

def handle_missing_values(df, threshold=0.5):
    """Remove columns with a high proportion of missing values and handle remaining nulls."""
    # Drop columns with more than `threshold` proportion of null values
    df = df.dropna(axis=1, thresh=int((1 - threshold) * len(df)))
    # Drop rows with any remaining null values
    df.dropna(inplace=True)
    return df

def preprocess_data(data, target_column):
    """Prepare the dataset for model training."""
    y = data[target_column].apply(lambda x: 0 if x == 'good' else 1)
    X = data.drop(columns=[target_column])
    return X, y
