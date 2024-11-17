import pandas as pd
from src.logger import get_logger

logger = get_logger(__name__)

def preprocess_data(data_path, target_col):
    """Load data, clean it, and prepare it for modeling."""
    logger.info("Starting data preprocessing...")
    try:
        # Load the dataset
        data = pd.read_csv(data_path)
        logger.info(f"Data loaded from {data_path} with shape {data.shape}")
        
        # Identify and drop columns with too many missing values
        missing_threshold = 0.5
        cols_to_drop = [col for col in data.columns if data[col].isnull().mean() > missing_threshold]
        logger.info(f"Columns to drop due to missing values: {cols_to_drop}")
        data.drop(columns=cols_to_drop, inplace=True)

        # Drop rows with any remaining missing values
        data.dropna(inplace=True)
        logger.info(f"Data shape after dropping missing values: {data.shape}")

        # Separate features and target
        X = data.drop(columns=[target_col])
        y = data[target_col]
        logger.info(f"Preprocessing completed. Feature set shape: {X.shape}, Target shape: {y.shape}")

        return X, y

    except Exception as e:
        logger.error(f"Error during preprocessing: {e}", exc_info=True)
        raise
