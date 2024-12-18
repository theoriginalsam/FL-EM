# utils/logging.py

import logging
import os
from datetime import datetime
from config.settings import RESULTS_DIR

class FLLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Create handlers
        self._setup_handlers()
        
    def _setup_handlers(self):
        # File handler
        log_file = os.path.join(RESULTS_DIR, f'fl_{datetime.now():%Y%m%d_%H%M}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatters and add it to handlers
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_metrics(self, metrics, round_num):
        self.logger.info(f"Round {round_num} Metrics:")
        for metric, value in metrics.items():
            self.logger.info(f"{metric}: {value}")

# utils/visualization.py
