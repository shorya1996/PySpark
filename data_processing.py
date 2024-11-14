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
    # Calculate the threshold for dropping columns
    cols_before = df.shape[1]
    drop_cols = df.columns[df.isnull().mean() > threshold]

    # Print the columns being dropped
    if len(drop_cols) > 0:
        print(f"Dropping columns due to high missing values: {list(drop_cols)}")
    else:
        print("No columns dropped due to missing values.")

    # Drop columns with more than `threshold` proportion of null values
    df.drop(columns=drop_cols, inplace=True)

    # Drop rows with any remaining null values
    rows_before = df.shape[0]
    df.dropna(inplace=True)
    rows_after = df.shape[0]

    print(f"Rows dropped due to null values: {rows_before - rows_after}")
    return df

def preprocess_data(data, target_column):
    """Prepare the dataset for model training."""
    y = data[target_column].apply(lambda x: 0 if x == 'good' else 1)
    X = data.drop(columns=[target_column])
    return X, y
