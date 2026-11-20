import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from src.logger import configure_logger
from src.exception import MyException
import sys
from src.processor.data_processor import data_processor, DataProcessing



logging = configure_logger()

class ModelTraining:
    def XGBModelTraining(self, X_train, y_train, preprocessor):
            try:
                # calculate scale pos weight for imbalance in xgboost classifier
                scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

                # create XGBoost model
                xgb_model = XGBClassifier(
                    random_state=42,
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    scale_pos_weight=scale_pos_weight,
                    eval_metrics='logloss' 
                )

                # Build the complete pipeline
                pipeline_xgb = Pipeline(
                    steps=[
                        ('preprocessor', preprocessor),
                        ('classifier', xgb_model)
                    ]
                )

                # Train the pipeline (using raw X_train, not processed)
                pipeline_xgb.fit(X_train, y_train)

                return pipeline_xgb
            except Exception as e:
                 logging.info(f"error occured while training this model {e}")
                 raise MyException(e, sys)

    def ModelEvaluation(self, trained_model, X_test, y_test):
         try:
            y_pred_xgb = trained_model.predict(X_test)
            model_accuracy = accuracy_score(y_pred=y_pred_xgb, y_true=y_test)
            model_precision_score = precision_score(y_pred=y_pred_xgb,y_true=y_test)
            model_recall_score = recall_score(y_pred=y_pred_xgb, y_true=y_test)
            model_f1_score = f1_score(y_pred=y_pred_xgb, y_true=y_test)

            logging.info(f"model has been successfully evaluated ..")
            return model_accuracy, model_precision_score, model_recall_score, model_f1_score
         except Exception as e:
              logging.error(f"error occured while evaluating the model {e}")
              raise MyException(e, sys)

def model_trainer():
    categorical_features, numerical_features, X_train, X_test, y_train, y_test = data_processor()
    DataProcessor = DataProcessing()

    preprocessor = DataProcessor.Data_processor_Engine(numerical_features=numerical_features, categorical_features=categorical_features)
    TrainingEngine = ModelTraining()
    Trained_Model = TrainingEngine.XGBModelTraining(X_train=X_train, y_train=y_train, preprocessor=preprocessor)
    logging.info(f"model has been successfully trained")
    model_accuracy, model_precision_score, model_recall_score, model_f1_score = TrainingEngine.ModelEvaluation(Trained_Model, X_test=X_test, y_test=y_test)

    logging.info(f"model accuracy = {model_accuracy}, model precision = {model_precision_score}, model recall = {model_recall_score}, model f1 score = {model_f1_score}")

    return Trained_Model, model_accuracy, model_precision_score, model_recall_score, model_f1_score


model_trainer() 


