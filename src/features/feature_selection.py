import os
from src.features.feature_engineering import feature_engineer
from src.logger import configure_logger
from src.exception import MyException

logging = configure_logger()
import sys

class Featureselection:
    def select_features(self, engineered_features):
        try:
            engineered_features = engineered_features.drop(
                ['transaction_id', 'customer_id', 'device_id', 'ip_address'], 
                axis=1
            )
            # Dropping some irrelevant variables
            engineered_features = engineered_features.drop(
                ['chargeback_history_count', 'exchange_rate_src_to_dest'], 
                axis=1
            )
            logging.info(f"feature have been selected and ready for modelling")
            logging.info(f"modelling data shape is (engineered_features.shape)")
            return engineered_features
        except Exception as e:
            logging.error(f"error occured while selecting features {e}")
            raise MyException(e, sys)

def feature_selector():
    engineered_data = feature_engineer()
    feature_selection_engine = Featureselection()
    modelling_data = feature_selection_engine.select_features(engineered_data)
    return modelling_data

#feature_selector()

