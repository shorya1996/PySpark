# src/logger.py
import logging
import uuid
import os

# Create a logger instance variable that will hold the logger
logger = None

def get_logger(name):
    """Create or get the global logger instance."""
    global logger
    if logger is None:
        # Generate a random UUID for the log file name
        log_id = str(uuid.uuid4())  # Generate a unique log ID
        log_filename = f'logs/project_{log_id}.log'  # Use uuid for log file name
        
        # Ensure the logs directory exists
        os.makedirs('logs', exist_ok=True)
        
        # Create a logger
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)  # Or any log level you prefer

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
