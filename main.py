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
        rf = RuleFit(tree_size=4, max_rule=2000, rfmode='classify', model_type="rl", randome_state=1, max_iter=1000)
        rf.fit(X,y,feature_names=X.columns)
        rules=rf.get_rules()
        print(rules)
        logger.info("Model training completed successfully")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()
    print(f"Logs saved to: logs/project_{logger.log_id}.log")
