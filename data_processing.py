import pandas as pd
import numpy as np

def load_data(file_path, target_col):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        print(f"Loaded data with shape: {data.shape}")
        return data
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None

def preprocess_data(df, target_col, drop_threshold=0.5):
    """Preprocess the dataset by handling null values and encoding."""
    # Replace placeholder 'none' values with NaN
    df.replace("none", np.NaN, inplace=True)

    # Drop columns with a high percentage of null values
    null_percentage = df.isnull().mean()
    cols_to_drop = null_percentage[null_percentage > drop_threshold].index
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"Dropped columns with > {drop_threshold*100}% null values: {list(cols_to_drop)}")

    # Drop rows with any remaining null values
    df.dropna(inplace=True)
    print(f"Shape after dropping rows with nulls: {df.shape}")

    # Split into features and target
    y = df[target_col].apply(lambda x: 0 if x == 'good' else 1)
    X = df.drop(target_col, axis=1)
    return X, y
