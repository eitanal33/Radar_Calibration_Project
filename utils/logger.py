import logging
import json
from logging import FileHandler, Formatter
from datetime import datetime

class JsonFormatter(Formatter):
    def format(self, record):
        log_record = {
            'level': record.levelname,
            'message': record.getMessage(),
            'time': self.formatTime(record, self.datefmt),
            'name': record.name,
        }
        return json.dumps(log_record)

def get_logger(log_file='logs.json'):
    logger = logging.getLogger('RadarCalibrationLogger')
    logger.setLevel(logging.INFO)

    # Check if the logger already has handlers to prevent duplicate logs
    if not logger.handlers:
        # Create a file handler
        file_handler = FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Set JSON formatter to the file handler
        file_handler.setFormatter(JsonFormatter())

        # Add the file handler to the logger
        logger.addHandler(file_handler)

    return logger

def log_experiment_header(logger, epochs):
    """
    Logs the experiment header with date, time, and number of epochs.
    """
    experiment_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header_info = {
        'experiment_start_time': experiment_start_time,
        'number_of_epochs': epochs,
    }
    logger.info(f"Experiment Info: {json.dumps(header_info)}")
