import os
import pandas as pd
import numpy as np
import sys

from src.logger import configure_logger
from src.exception import MyException
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from src.features.feature_selection import feature_selector

logging = configure_logger()

class DataProcessing:
    def column_checker(self, Modelling_Data):
        try:
            categorical_features = Modelling_Data.select_dtypes(include=['object', 'bool']).columns
            numerical_features = Modelling_Data.select_dtypes(include=['int', 'float']).columns.drop('is_fraud')
            all_features = list(categorical_features) + list(numerical_features)
            logging.info(f"categorical feature {len(categorical_features)}")
            logging.info(f"numerical features: {len(numerical_features)}")

            return categorical_features, numerical_features, all_features

        except Exception as e:
            logging.error(f"error occured while checking fro column data types {e}")
            raise MyException(e, sys)

    def DataSpliter(self, Modelling_Data, all_features):
        try:
            # sort by timestamp
            Modelling_Data = Modelling_Data.sort_values(by='timestamp')

            # 80/20 train_test split
            split_index = int(len(Modelling_Data) * 0.8) # This determines where the dataset should be split
            train_Modelling_Data = Modelling_Data.iloc[:split_index] # Training on feature  transactions
            test_Modelling_Data = Modelling_Data.iloc[split_index:]# testing on past transactions

            # preparing x and Y
            X_train = train_Modelling_Data[all_features]
            y_train = train_Modelling_Data['is_fraud']
            X_test = test_Modelling_Data[all_features]
            y_test = test_Modelling_Data['is_fraud']

            logging.info(f"\nX_train: {X_train.shape}, y_train: {y_train.shape}")
            logging.info(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

            return X_train, X_test, y_train, y_test

        except Exception as e:
            logging.error(f"error occured while splitting the data {e}")
            raise MyException(e, sys)

    def Data_processor_Engine(self, numerical_features, categorical_features):
        try:
            # Building a preprocessing pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_features),
                    ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
                ]
            )
            return preprocessor
        except Exception as e:
            logging.error(f"error occured while processing the data {e}")
            raise MyException(e, sys)


def data_processor():
    Modelling_Data = feature_selector()
    data_processing_engine = DataProcessing()
    categorical_features, numerical_features, all_features = data_processing_engine.column_checker(Modelling_Data)
    X_train, X_test, y_train, y_test = data_processing_engine.DataSpliter(Modelling_Data, all_features)

    return categorical_features, numerical_features, X_train, X_test, y_train, y_test

#data_processor()

