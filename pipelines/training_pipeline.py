from zenml import pipeline

from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluate import evaluate_model

@pipeline
def train_pipeline(data_path: str):
    df = ingest_df(data_path = data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train, y_train)
    mse, rmse, r2_score  = evaluate_model(X_test, y_test, model)