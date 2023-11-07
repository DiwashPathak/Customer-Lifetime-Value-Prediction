import logging
from typing import Tuple
from typing_extensions import Annotated
import pandas as pd

import mlflow
from sklearn.base import RegressorMixin
from zenml import step 
from zenml.client import Client

from src.evaluation import MSE, RMSE, R2

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker = experiment_tracker.name)
def evaluate_model(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model: RegressorMixin
)-> Tuple[
    Annotated[float, 'MSE'],
    Annotated[float, 'RMSE'],
    Annotated[float, 'R2_score']
]:
    """ 
    Calculates evaluation metrics of the model
    
    Args:
        X_test: Testing data
        y_test: Testing labels

    Returns:
        mse: mean squared error
        rmse: root mean squared error 
        r2_score: R Squared
    """
    try:
        y_pred = model.predict(X_test)

        # Mean Squared Error
        mse = MSE().calculate_scores(y_true =y_test, y_pred =y_pred)
        mlflow.log_metric('MSE',mse)

        # Root Mean Squared Error
        rmse = RMSE().calculate_scores(y_true =y_test, y_pred= y_pred)
        mlflow.log_metric('RMSE',rmse)

        # R2 Score
        r2_score = R2().calculate_scores(y_true =y_test, y_pred= y_pred)
        mlflow.log_metric('R2_score',r2_score)

        return mse, rmse, r2_score
    
    except Exception as e:
        logging.error(f'Error in evaluate.py {e}')
        raise e