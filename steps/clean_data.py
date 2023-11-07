from zenml import step 
import pandas as pd
from typing import Annotated
from typing_extensions import Tuple

from src.data_cleaning import (DataPreProcessStrategy, DataDivideStrategy, DataCleaning)

@step(enable_cache=False)
def clean_df(df: pd.DataFrame)->Tuple[
    Annotated[pd.DataFrame, 'X_train'],
    Annotated[pd.DataFrame, 'X_test'],
    Annotated[pd.Series, 'y_train'],
    Annotated[pd.Series, 'y_test']
]:
    """
    Preprocessing raw data and dividing it into train and test

    Args:
        df: Data to be handled.
    
    Returns:
        X_train(pd.DataFrame): Training features.
        X_test(pd.DataFrame) : Testing features.
        y_train(pd.Series) : Training labels
        y_test(pd.Series)  : Testing labels
    """
    # Preprocessing the data
    preprocess_strategy = DataPreProcessStrategy()
    data_cleaning = DataCleaning(df = df, strategy = preprocess_strategy)
    preprocessed_data = data_cleaning.handle_data()

    # Dividing the data
    data_divide = DataDivideStrategy()
    data_cleaning = DataCleaning(df = preprocessed_data, strategy = data_divide)
    X_train, X_test, y_train, y_test = data_cleaning.handle_data()
    return X_train, X_test, y_train, y_test



