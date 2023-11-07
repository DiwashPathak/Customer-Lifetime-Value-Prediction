from zenml import step
import pandas as pd
import logging

class IngestData:
    """ Ingests the data from the given data path"""
    def __init__(self, data_path: str):
        """
        Args:
        `   data_path: Path to the data
        """
        self.data_path = data_path
    
    def get_data(self):
        """
        Loads the data and returns a dataframe
        
        Returns:
            df(pd.DataFrame): The ingested data.
        """
        logging.info(f'Ingesting data from: {self.data_path}')
        df = pd.read_csv(self.data_path)
        return df

@step
def ingest_df(data_path: str)-> pd.DataFrame:
    """
    Ingesting data from data_path

    Args:
        data_path: Path of data to be ingested.
    
    Returns:
        df(pd.DataFrame): The ingested data
    """
    try:
        ingest_df = IngestData(data_path = data_path)
        df = ingest_df.get_data()
        return df
    
    except Exception as e:
        logging.error(f'Error occured while ingesting data from: {e}')
        raise e