import logging
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    """ Abstract class defining strategy for evaluating our models """
    @abstractmethod
    def calculate_scores(self, y_true:np.ndarray, y_pred:np.ndarray):
        """
        Calculates the score of the model

        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        pass

class MSE(Evaluation):
    """ Evaluation strategy that uses Mean Squared Error"""
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray)-> np.number:
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f'MSE: {mse}')
            return mse
        
        except Exception as e:
            logging.error(f"Error in calcualting MSE: {e}")
            raise e

class R2(Evaluation):
    """ Evaluation Strategy that uses R2 Score."""
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating R2 Schore")
            r2 = r2_score(y_true, y_pred)
            logging.info(f'R2: {r2}')
            return r2 
        
        except Exception as e:
            logging.error(f"Error in calcualting R2 Score {e}")
            raise e
        
class RMSE(Evaluation):
    """ Evaluation Strategy that uses Root Mean Squared Error."""
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
       try:
           logging.info("Calculating MSE")
           rmse = np.sqrt(mean_squared_error(y_true, y_pred))
           logging.info(f'RMSE: {rmse}')
           return rmse
       
       except Exception as e:
           logging.error(f"Error in calculating MSE: {e}")
           raise e