import logging
import pandas as pd

from .config import ModelNameConfig
from src.model_dev import LinearRegressionModel 

import mlflow
from zenml import step
from sklearn.base import RegressorMixin
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker = experiment_tracker.name)
def train_model( 
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    config: ModelNameConfig
)-> RegressorMixin:
    """
    Trains the model on the training set

    Args:
        X_train: Training data
        y_train: Training labels

    Returns:
        trained_model
    """
    try:
        if config.model_name == 'LinearRegression':
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
        else:
            raise ValueError(f'Model {config.model_name} is not supported!')
    
    except Exception as e:
        logging.error(f'Error occured in training model {e}')