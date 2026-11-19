import os
import pandas as np
import numpy as np

from src.data.data_cleaning import DataCleaner
from src.logger import configure_logger
from src.exception import MyException
import sys
logging = configure_logger()

class DataValidation():
    def checking_negative_values(self, customer_transaction_data):
        try:
            # checking negative values in numeric columns
            negative_counts = {
                'amount_src': (customer_transaction_data['amount_src'] < 0).sum(),
                'amount_usd': (customer_transaction_data['amount_usd'] < 0).sum(),
                'fee': (customer_transaction_data['fee']<0).sum(),
                'device_trust_score': (customer_transaction_data['device_trust_score']<0).sum(),
                'txn_velocity_1h': (customer_transaction_data['txn_velocity_1h'] <0).sum(),
                'txn_velocity_24h': (customer_transaction_data['txn_velocity_24h'] <0).sum(),
                'risk_score_internal': (customer_transaction_data['risk_score_internal'] < 0).sum(),
            }
            logging.info(f"error with the negative values {negative_counts}")

        except Exception as e:
            logging.error(f"error occured while checking fro negative values")
            raise MyException(e, sys)
    
    def dropping_negative_values(self, customer_transaction_data):
        try:
            customer_transaction_data = customer_transaction_data.drop(customer_transaction_data[
            (customer_transaction_data['amount_src'] <0) |
            (customer_transaction_data['fee'] < 0) |
            (customer_transaction_data['device_trust_score'] < 0) |
            (customer_transaction_data['txn_velocity_1h'] <0)].index)
            logging.info(f"negative values have been dropped successfully")

            return customer_transaction_data
        except Exception as e:
            logging.error(f"error occured while dropping negative values {e}")
            raise MyException(e, sys)

def DataValidator():
    try:
        cleaned_customer_data = DataCleaner()
        data_validation_engine = DataValidation()
        data_validation_engine.checking_negative_values(cleaned_customer_data)
        cleaned_validated_customer_data = data_validation_engine.dropping_negative_values(cleaned_customer_data)

        return cleaned_validated_customer_data
    except Exception as e:
        logging.error(f"error occured while validating a data {e}")
        raise MyException(e, sys)

DataValidator()

