import logging 
from abc import ABC, abstractmethod

import pandas as pd
from sklearn.linear_model import LinearRegression

class Model(ABC):
    """ Abstract base for all models"""
    @abstractmethod
    def train(self, X_train:pd.DataFrame, y_train:pd.Series):
        """ Trains the model"""
        pass

class LinearRegressionModel(Model):
    """ Linear Regression Model"""
    def train(self, X_train: pd.DataFrame, y_train: pd.Series)-> LinearRegression:
        """
        Trains the linear regression model

        Args:
            X_train: Training data
            y_train: Training labels
        
        Returns:
            reg_model: Trained linear regression model
        """
        try:
            reg_model = LinearRegression()
            reg_model.fit(X_train, y_train)
            logging.info('Model training completed.')
            return reg_model
        except Exception as e:
            logging.error(f'Error in training model {e}')
            raise e