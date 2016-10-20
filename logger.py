__author__ = 'pralav'
from project_settings import LOGGING_CONFIG
import os
import json
import logging.config

def setup_logging(
        default_path=LOGGING_CONFIG,
        default_level=logging.INFO,
        env_key='LOG_CFG',save_path=None
):
    """Setup logging configuration

    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
        if save_path is not None:
            config['handlers']['info_file_handler']['filename']=save_path
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
    return logging
