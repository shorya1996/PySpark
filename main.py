from src.logger import get_logger
from src.preprocessing import preprocess_data
from src.model import train_model

logger = get_logger(__name__)

def main():
    logger.info("Starting the end-to-end pipeline")
    data_path = "data/input.csv"
    target_col = "Class"

    try:
        X, y = preprocess_data(data_path, target_col)
        logger.info("Data preprocessing completed successfully")
        model = train_model(X, y)
        logger.info("Model training completed successfully")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()
    print(f"Logs saved to: logs/project_{logger.log_id}.log")
