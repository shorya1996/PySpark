# src/logger.py
import logging
import uuid
import os

def get_logger(name):
    """Creates a logger that writes to a unique log file generated using uuid."""
    
    # Generate a random UUID for the log file name
    log_id = str(uuid.uuid4())  # Generate a unique log ID
    log_filename = f'logs/project_{log_id}.log'  # Use uuid for log file name
    
    # Ensure the logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)  # Or any log level you prefer
    
    # Check if the logger already has handlers, if not, add a new one
    if not logger.handlers:
        # Create file handler to log to a file
        file_handler = logging.FileHandler(log_filename)  # Unique log file
        file_handler.setLevel(logging.INFO)

        # Create a formatter and set it for the file handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(file_handler)

    # Adding the log ID to logger so it can be accessed
    logger.log_id = log_id
    
    return logger
