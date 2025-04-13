import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_params import *
from utils.common_functions import read_yaml, load_data
from scipy.stats import randint
##For experiment tracking we use ML-flow
# MLflow is a platform to manage the ML lifecycle, including experimentation, reproducibility, and deployment.
import mlflow
import mlflow.sklearn

logger = get_logger(__name__)

class ModelTraining:

    def __init__(self, train_path, test_path, model_output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path
        
        self.params_dist = LIGHTGBM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS


    def load_and_split_data(self):
        try:
            logger.info(f"Loading training data from {self.train_path} and test data from {self.test_path}")
            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            X_train = train_df.drop(columns=['booking_status'])
            y_train = train_df['booking_status']

            X_test = test_df.drop(columns=['booking_status'])
            y_test = test_df['booking_status']

            logger.info("Data loaded and split successfully.")

            return X_train, y_train, X_test, y_test
        
        except Exception as e:
            logger.error(f"Error in loading or splitting data: {e}")
            raise CustomException('Failed to load data', e)
        
    def train_lgbm(self, X_train, y_train):
        try:
            logger.info("Initialising LightGBM model")
            
            lgbm_model = lgb.LGBMClassifier(random_state=self.random_search_params['random_state'])
            
            logger.info("Starting Randomized Search for hyperparameter tuning")

            random_search = RandomizedSearchCV(
                estimator=lgbm_model,
                param_distributions=self.params_dist,
                n_iter=self.random_search_params['n_iter'],
                cv=self.random_search_params['cv'],
                verbose=self.random_search_params['verbose'],
                n_jobs=self.random_search_params['n_jobs'],
                random_state=self.random_search_params['random_state'],
                scoring=self.random_search_params['scoring']
            )
            
            logger.info("Starting Hyperparameter tuning.")
            
            random_search.fit(X_train, y_train)

            logger.info("Hyperparameter tuning completed.")

            best_params = random_search.best_params_
            best_lgbm_model = random_search.best_estimator_

            logger.info(f"Best parameters found: {best_params}")

            return best_lgbm_model
        
        except Exception as e:
            logger.error(f"Error in training the model: {e}")
            raise CustomException('Failed to train the model', e)
        

    def evaluate_model(self, model, X_test, y_test):
        try:
            logger.info("Evaluating the model")
            
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            logger.info(f"Model evaluation metrics: Accuracy: {accuracy}, Recall: {recall}, Precision: {precision}, F1 Score: {f1}")

            return {
                'accuracy' : accuracy,
                'recall' : recall,
                'precision' : precision,
                'f1' : f1            
            }
        
        except Exception as e:
            logger.error(f"Error in evaluating the model: {e}")
            raise CustomException('Failed to evaluate the model', e)
        

    def save_model(self, model):
        try:
            logger.info(f"Saving the model to {self.model_output_path}")
            
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)

            joblib.dump(model, self.model_output_path)

            logger.info(f"Model saved successfully to {self.model_output_path}.")
        
        except Exception as e:
            logger.error(f"Error in saving the model: {e}")
            raise CustomException('Failed to save the model', e)


    def run(self):
        try:
            with mlflow.start_run():
                logger.info("Starting the model training pipeline")

                logger.info("Starting MLflow experimentation")

                logger.info("Logging the training and testing datasets to MLflow")
                mlflow.log_artifact(self.train_path, artifact_path="datasets")
                mlflow.log_artifact(self.test_path, artifact_path="datasets")
                
                X_train, y_train, X_test, y_test = self.load_and_split_data()
                
                best_lgbm_model = self.train_lgbm(X_train, y_train)
                
                metrics = self.evaluate_model(best_lgbm_model, X_test, y_test)
                
                self.save_model(best_lgbm_model)

                logger.info("Logging model to MLflow")
                mlflow.log_artifact(self.model_output_path, artifact_path="models")

                logger.info("Logging metrics and params to MLflow")
                mlflow.log_metrics(metrics)
                mlflow.log_params(best_lgbm_model.get_params())

                logger.info("Model training process completed successfully.")
            

        
        except Exception as e:
            logger.error(f"Error in the model training process: {e}")
            raise CustomException('Failed to run the model training process', e)
        

if __name__ == "__main__":
    # Example usage
    model_trainer = ModelTraining(PROCESSED_TRAIN_DATA_PATH, PROCESSED_TEST_DATA_PATH, MODEL_OUTPUT_PATH)

    model_trainer.run()