import os
import pandas as pd
import numpy as np
from config.constant import DATA_DIR
from src.logger import configure_logger
import sys
from src.exception import MyException

logging = configure_logger()

def load_data():
    try:
        # Loading the dataset
        customer_transaction_data = pd.read_csv(DATA_DIR)
        logging.info(f"data has been successfully loaded...")
        logging.info(f"{customer_transaction_data.head()}")
        return customer_transaction_data
    except FileNotFoundError:
        logging.info(f"File Not founf at {DATA_DIR}, please check data file path")
        return None
    except Exception as e:
        logging.error(f"an error occurred while loading the data {e}")
        raise MyException(e, sys)

#load_data()