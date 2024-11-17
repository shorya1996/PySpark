import logging
import os
import uuid
from datetime import datetime

def get_logger(name):
    """Configure and return a logger with a unique log file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Generate a unique log ID for each session
    log_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
    log_file = f"logs/project_{log_id}.log"

    # Ensure the logs directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Create a file handler to log to a unique file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Create a console handler for real-time logging
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Define the logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    logger.propagate = False

    # Attach log_id for reference in the logger
    logger.log_id = log_id

    return logger
