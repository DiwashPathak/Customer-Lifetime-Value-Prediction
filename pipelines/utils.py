import logging
import pandas as pd
from src.data_cleaning import DataCleaning, DataPreProcessStrategy

def get_data_for_test():
    try:
        df = pd.read_csv('/home/diwas/Documents/customer lifetime value/customer lifetime value/data/WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv')
        preprocess_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df , preprocess_strategy)
        df = data_cleaning.handle_data()
        df = df.sample(n = 100)
        df.drop(['Customer Lifetime Value'], axis = 1, inplace = True)
        result = df.to_json(orient = 'split')
        return result
    
    except Exception as e:
        logging.error(f'Error occured in get_data_for_test {e}')
        raise e