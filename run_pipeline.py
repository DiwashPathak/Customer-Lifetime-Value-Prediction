from pipelines.training_pipeline import train_pipeline
from zenml.client import Client

if __name__ == '__main__':
    print(f'Tracking uri:{Client().active_stack.experiment_tracker}')
    train_pipeline(data_path = '/home/diwas/Documents/customer lifetime value/customer lifetime value/data/WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv')
    