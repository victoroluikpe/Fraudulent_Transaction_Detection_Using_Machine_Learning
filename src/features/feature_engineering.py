import pandas as pd
import numpy as np
from src.logger import configure_logger
from src.exception import MyException
from src.data.data_validation import DataValidator
import sys

logging = configure_logger()

class featureEngineering:
    def engine_feature(self, cleaned_customer_transaction_data):
        try:
            cleaned_customer_transaction_data['timestamp'] = pd.to_datetime(cleaned_customer_transaction_data['timestamp'])
            # Extract time-based features
            cleaned_customer_transaction_data['hour'] = cleaned_customer_transaction_data['timestamp'].dt.hour
            cleaned_customer_transaction_data['day_of_week'] = cleaned_customer_transaction_data['timestamp'].dt.dayofweek
            cleaned_customer_transaction_data['is_weekend'] = (cleaned_customer_transaction_data['day_of_week'] >=5).astype(int)
            cleaned_customer_transaction_data['month'] = cleaned_customer_transaction_data['timestamp'].dt.month
            cleaned_customer_transaction_data['late_night_hours'] = ((cleaned_customer_transaction_data['hour'] >= 3) & (cleaned_customer_transaction_data['hour'] <= 7)).astype(int)
            cleaned_customer_transaction_data['amount_high'] = (cleaned_customer_transaction_data['amount_usd'] >= 1000).astype(int)
            cleaned_customer_transaction_data['high_ip_risk'] = (cleaned_customer_transaction_data['ip_risk_score'] >0.8).astype(int)
            cleaned_customer_transaction_data['low_device_trust'] = (cleaned_customer_transaction_data['device_trust_score'] <0.5).astype(int)
            cleaned_customer_transaction_data['new_account'] = ((cleaned_customer_transaction_data['account_age_days'] >=30)&(cleaned_customer_transaction_data['account_age_days'] < 90)).astype(int) 
            cleaned_customer_transaction_data['very_new_account'] = (cleaned_customer_transaction_data['account_age_days'] < 30).astype(int)
            cleaned_customer_transaction_data['velocity_spike'] = (cleaned_customer_transaction_data['txn_velocity_1h'] >= 3).astype(int)

            logging.info(f"cleaned customer data shape {cleaned_customer_transaction_data.shape}")
            logging.info(f"data has been successfully engineered..")
            return cleaned_customer_transaction_data
        except Exception as e:
            logging.error(f"cleaned customer data shape {cleaned_customer_transaction_data.shape}")
            raise MyException(e, sys)


def feature_engineer():
    cleaned_validated_customer_data = DataValidator()
    feature_engineering_engine = featureEngineering()
    engineered_data = feature_engineering_engine.engine_feature(cleaned_validated_customer_data)

    return engineered_data

feature_engineer()
