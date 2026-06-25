import os
import logging
from datetime import datetime

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

class DbLoggingHandler(logging.Handler):
    """Custom logging handler to write logs to the database system_logs table."""
    def __init__(self, db_service=None):
        super().__init__()
        self.db_service = db_service

    def emit(self, record):
        if self.db_service is None:
            return
        try:
            log_entry = self.format(record)
            self.db_service.log_to_db(
                level=record.levelname,
                module=record.module,
                message=record.getMessage()
            )
        except Exception:
            # Avoid infinite loops or crashes during logging
            pass

# Configure python logging
logger = logging.getLogger("mide")
logger.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler("data/system_logs.log", encoding="utf-8")
file_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(module)s: %(message)s"
)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(file_formatter)
logger.addHandler(console_handler)

# Database handler reference to be populated later
db_handler = DbLoggingHandler()
logger.addHandler(db_handler)

def get_logger():
    """Return the system logger."""
    return logger

def set_db_service(db_service):
    """Set db_service on the DB logging handler to enable database logging."""
    db_handler.db_service = db_service
