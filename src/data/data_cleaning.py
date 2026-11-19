import pandas as pd
import numpy as np
from src.data.data_ingestion import load_data
from src.logger import configure_logger
import sys

from src.exception import MyException

logging = configure_logger()

class DataCleaning:
    def check_data_structure(self, customer_transaction_data):
        # checking for all the columns in the dataset
        logging.info(f"{customer_transaction_data.columns}")
        return customer_transaction_data

    def checking_for_missingvalues(self, customer_transaction_data):
        logging.info(f"columns_with missing values {customer_transaction_data.isna().sum()}")
        return customer_transaction_data

    def formatting_data(self, customer_transaction_data):
        try:
            # converting data types of timestamp to datetime
            customer_transaction_data['timestamp'] = pd.to_datetime(customer_transaction_data['timestamp'],errors='coerce')

            # converting data type of income_usd to float
            customer_transaction_data['amount_src'] = pd.to_numeric(customer_transaction_data['amount_src'], errors='coerce')
            return customer_transaction_data
        except Exception as e:
            logging.error(f"error occured while formatting the data {e}") 
            raise MyException(e, sys)

    def exchange_rate_calculator(self, customer_transaction_data):
        try:
            # calculating exchange rates per currency 
            exchange_rates = customer_transaction_data[customer_transaction_data['amount_usd'].notna()].groupby('source_currency').apply(
            lambda x: (x['amount_usd'] / x['amount_src']).mean()
            ).to_dict()
            logging.info(f"exchange_rate has been sucessfully created: {exchange_rates}")
            # filling the missing values of the amount_usd
            customer_transaction_data['amount_usd']=customer_transaction_data.apply(
                lambda row: row['amount_usd'] if pd.notna(row['amount_usd']) else row['amount_src'] * exchange_rates.get(row['source_currency'],1),
                axis=1
            )
            logging.info(f"amount i usd have been automatically be filled ...")
            return customer_transaction_data

        except Exception as e:
            logging.error(f"error occured while calculating the exchange rate")
            raise MyException(e, sys)

    def filling_missing_values(self, customer_transaction_data):
        try:
            # filling the missing values of the fee variable
            if 'fee' in customer_transaction_data.columns:
                if 'channel' in customer_transaction_data.columns:
                    customer_transaction_data['fee'] = customer_transaction_data.groupby('channel')['fee'].transform(lambda y: y.fillna(y.median()))
                customer_transaction_data['fee'] = customer_transaction_data['fee'].median()

                # filling the missing values of ip_country variable
            if {'ip_country', 'home_county'}.issubset(customer_transaction_data.columns):
                customer_transaction_data['ip_country'] = customer_transaction_data['ip_country'].fillna(customer_transaction_data['home_country'])

                # filling the missing values with device_trust_score variable
            if 'device_trust_score' in customer_transaction_data.columns:
                if{'new_device', 'kyc_tier'}.issubset(customer_transaction_data.columns):
                    customer_transaction_data['device_trust_score']=customer_transaction_data.groupby(['new_device', 'kyc_tier'])['device_trust_score'].transform(lambda x: x.fillna(x.median()))
                customer_transaction_data['device_trust_score']=customer_transaction_data['device_trust_score'].fillna(customer_transaction_data['device_trust_score'].median())

            customer_transaction_data.dropna(subset=['timestamp', 'amount_src', 'ip_address'], inplace = True)
            logging.info(f"data has been sucessfully cleaned")
            logging.info(f"{customer_transaction_data.isna().sum()}")

            return customer_transaction_data
        except Exception as e:
            logging.error(f"error occured while filling the values {e}")
            raise MyException(e, sys)

    def normalizing_data(self, customer_transaction_data):
        try:
            # reviewing the channel varable
            customer_transaction_data['channel'].unique()
            # formatting the channel variable
            customer_transaction_data['channel'] = customer_transaction_data['channel'].str.lower().str.strip()
            customer_transaction_data['channel'].unique()

            # we try to hamornize the mis-spelled
            customer_transaction_data['channel'] = customer_transaction_data['channel'].replace({
                'weeb':'web',
                'mobille':'mobile',
            })
            customer_transaction_data['channel'] = customer_transaction_data['channel'].replace({'unknown':np.nan})
            customer_transaction_data['channel'].unique()
            # review the kyc_tier variable
            customer_transaction_data['kyc_tier'].unique()
            customer_transaction_data['kyc_tier'] = customer_transaction_data['kyc_tier'].replace({
            'standrd':'standard',
            'enhancd':'enhanced',
            'unknown':'np.nan'
        })
            customer_transaction_data['kyc_tier'].unique()
            customer_transaction_data.dropna(inplace=True)
            customer_transaction_data.info()
            # reviewing the home_country variable
            customer_transaction_data['home_country'].unique()
            #formatting the home_country variable
            customer_transaction_data['home_country'] = customer_transaction_data['home_country'].str.lower().str.strip()
            customer_transaction_data['home_country'].unique()
            # reviewing the ip_country variable
            customer_transaction_data['ip_country'].unique()
            # formatting the ip_country variables
            customer_transaction_data['ip_country'] = customer_transaction_data['ip_country'].str.lower().str.strip()
            customer_transaction_data['ip_country'].unique()

            logging.info(f"dataset has been successfully normalized")

            return customer_transaction_data

        except Exception as e:
            logging.error(f"error occured while normalizing the dataset {e}")
            raise MyException(e, sys)

def DataCleaner():
    data = load_data()
    data_cleaning_engine = DataCleaning()
    data = data_cleaning_engine.check_data_structure(data)
    data = data_cleaning_engine.checking_for_missingvalues(data)
    formatted_data = data_cleaning_engine.formatting_data(data)
    data = data_cleaning_engine.exchange_rate_calculator(formatted_data)
    cleaned_customer_data = data_cleaning_engine.filling_missing_values(data)
    normalized_customer_data = data_cleaning_engine.normalizing_data(cleaned_customer_data)

    logging.info(f"{normalized_customer_data.isna().sum()}")

    return normalized_customer_data

#DataCleaner()        
