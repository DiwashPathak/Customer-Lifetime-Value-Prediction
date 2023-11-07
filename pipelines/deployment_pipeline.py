import logging
import numpy as np
import pandas as pd
from zenml import pipeline, step
from typing_extensions import  Annotated
import json

from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer

from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.evaluate import evaluate_model
from steps.model_train import train_model
from .utils import get_data_for_test

docker_settings = DockerSettings(required_integrations=[MLFLOW])

class DeploymentTriggerConfig(BaseParameters):
    " Deployment trigger config"
    min_accuracy = 0.0

@step(enable_cache = False)
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig
):
    """ Implementes deployment trigger that looks at the model input model accuarcy and gives deployment decision"""
    return accuracy >= config.min_accuracy


@pipeline(enable_cache = False, settings = {'docker': docker_settings})
def continuous_deployment_pipeline(
        data_path:str,
        min_accuracy: float = 0.0,
        workers:int = 1,
        timeout:int = DEFAULT_SERVICE_START_STOP_TIMEOUT
):
    """ Implements continuous deployment pipeline"""
    df = ingest_df(data_path = data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train, y_train)
    mse, rmse, r2_score  = evaluate_model(X_test, y_test, model)
    deployment_decision = deployment_trigger(accuracy = r2_score)
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout,
    )


@step(enable_cache = False)
def dynamic_importer()->str:
    """ Dynamic importer"""
    data = get_data_for_test()
    return data

@step(enable_cache = False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running:bool = True,
    model_name: str = 'model'
) -> MLFlowDeploymentService:
    """ 
    Gets the prediction server deployed by deployment pipeline
    
    Args:
        pipeline_name: The name of the pipeline that deployed the MLflow prediction server
        step_name: The name of the step that deployed the MLflow prediction server
        running: When this flag is set, the step only returns a running service
        model_name: The name of the model that is deployed
    """
    # Get the MLflow model deployer stack component
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    # fetch the exisisting services with same pipeline, step & model
    existing_services = model_deployer.find_model_server(
        pipeline_name = pipeline_name,
        pipeline_step_name= pipeline_step_name,
        model_name= model_name,
        running = running
    )
    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{pipeline_step_name} step in the {pipeline_name} "
            f"pipeline for the '{model_name}' model is currently "
               f"running."
        )
    return existing_services[0]
    
@step(enable_cache= False)
def predictor(
    service: MLFlowDeploymentService,
    data: str
) ->np.ndarray:
    """ Run an inference against the prediction service"""
    service.start(timeout=10)
    data = json.loads(data)
    columns = data.pop('columns')
    data.pop('index')
    df = pd.DataFrame(data['data'], columns = columns)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    prediction = service.predict(df)
    return prediction

@pipeline(enable_cache = False, settings = {'docker': docker_settings})
def inference_pipeline(
    pipeline_name:str,
    pipeline_step_name: str
):
    """ Links all of the steps artifacts """
    data = dynamic_importer()
    deployment_service = prediction_service_loader(
        pipeline_name = pipeline_name,
        pipeline_step_name = pipeline_step_name,
        running = False
    )
    prediction = predictor(
        service = deployment_service,
        data = data
    )