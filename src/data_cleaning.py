import logging
from typing import Union
from typing_extensions import Tuple
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """ Abstract class for defining strategy to handle data"""
    @abstractmethod
    def handle_data(self, df: pd.DataFrame)-> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreProcessStrategy(DataStrategy):
    """ DataPreProcessing strategy that preprocesses raw data"""
    def handle_data(self, df: pd.DataFrame)-> pd.DataFrame:
        """
        Preprocessing the data.

        Args:
            df: Raw data to be processed
        """
        try:
            # Removing the columns that are not required
            cols_to_drop = ['Customer', 'State', 'Location Code','Effective To Date', 'Renew Offer Type', 'Policy']
            df = df.drop(columns = cols_to_drop , axis = 1)
    
            # Tunning binary colums
            df['Response'] = df['Response'].apply(lambda x: 1 if 'Yes' else 'No')
            df['EmploymentStatus'] = df['EmploymentStatus'].apply(lambda x: 1 if 'Employed' else 0)
            df['Gender'] = df['Gender'].apply(lambda x: 1 if 'M' else 0)
            df['Marital Status'] = df['Marital Status'].apply(lambda x:1 if 'Married' else 0)
    
            # Encoding nominal categorical columns
            nominal_cols = df[['Policy Type', 'Sales Channel', 'Vehicle Class']]
            encoded_nominal = pd.get_dummies(nominal_cols, drop_first = True)
    
            # Encoding of ordinal categorical columns
            ordinal_cols = df[['Coverage', 'Education', 'Vehicle Size']]
            ordinal_enc = OrdinalEncoder(categories = [['Basic', 'Extended', 'Premium'], ['High School or Below', 'Bachelor', 'College', 'Master', 'Doctor'], ['Small', 'Medsize', 'Large']])
            encoded_ordinal = ordinal_enc.fit_transform(ordinal_cols)
            encoded_ordinal = pd.DataFrame(encoded_ordinal, columns = ordinal_enc.get_feature_names_out())
    
            # Get all of the numeric columns
            numeric_cols = df[['Customer Lifetime Value','Income','Monthly Premium Auto','Months Since Last Claim','Months Since Policy Inception','Number of Open Complaints','Number of Policies','Total Claim Amount','Response', 'EmploymentStatus', 'Gender', 'Marital Status']]
    
            # Concat numerical and encoded columns
            df = pd.concat([numeric_cols, encoded_nominal, encoded_ordinal], axis = 1)
            return df
        
        except Exception as e:
            logging.error(f'Error in preprocessing data {e}')
            raise e

class DataDivideStrategy(DataStrategy):
    """ Divides the into train and test"""
    def handle_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """ 
        Divides the data into trani and test

        Args:
            df: The data to be divided

        Returns:
            X_train(pd.DataFrame): Training features.
            X_test(pd.DataFrame) : Testing features.
            y_train(pd.Series) : Training labels
            y_test(pd.Series)  : Testing labels
        """
        try:
            X = df.drop(columns = ['Customer Lifetime Value'])
            y  =df['Customer Lifetime Value']
    
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            logging.error(f'Error in dividing data {e}')
            raise e

class DataCleaning:
    """ DataCleaning class that implements other strategies"""
    def __init__(self, df: pd.DataFrame , strategy: DataStrategy):
        """ Initializes DataCleaning class"""
        self.data = df
        self.strategy = strategy
    
    def handle_data(self):
        return self.strategy.handle_data(self.data)