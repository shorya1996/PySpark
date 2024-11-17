# src/logger.py
import logging
import uuid
import os

# Global logger variable to hold the logger instance
logger = None

def get_logger(name):
    """Create or get the global logger instance."""
    global logger
    if logger is None:
        # Generate a unique UUID for the session's log file name
        log_id = str(uuid.uuid4())  # Generate a unique log ID for the session
        log_filename = f'logs/project_{log_id}.log'  # Use UUID in log file name
        
        # Ensure that the logs directory exists
        os.makedirs('logs', exist_ok=True)
        
        # Create a global logger that will be used by all modules
        logger = logging.getLogger()  # Root logger
        logger.setLevel(logging.DEBUG)  # Set log level to DEBUG to capture all logs

        # Create a file handler to write logs to the file
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.DEBUG)  # Set file handler to capture DEBUG level logs

        # Define the format for the log entries
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add the file handler to the logger
        logger.addHandler(file_handler)

        # Debugging: Confirm the logger initialization and file handler
        logger.debug(f"Logger initialized with log file: {log_filename}")

        # Store the log ID for reference in the log file name
        logger.log_id = log_id

    # Get a logger for the specific module, sharing the same file handler
    module_logger = logging.getLogger(name)

    # Ensure the module logger propagates its logs to the global logger
    module_logger.propagate = True
    
    return module_logger
