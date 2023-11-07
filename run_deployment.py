import click
from typing import cast
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from pipelines.deployment_pipeline import continuous_deployment_pipeline, inference_pipeline

DEPLOY = 'deploy'
PREDICT = 'predict'
DEPLOY_AND_PREDICT = 'deploy_and_predict'

@click.command()
@click.option(
    '--config',
    type = click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    help = "Optionally you can choose to only run the deployment "
    "pipeline to train and deploy a model (`deploy`), or to "
    "only run a prediction against the deployed model "
    "(`predict`). By default both will be run "
    "(`deploy_and_predict`)."
)
@click.option(
    '--min-accuracy',
    default = 0.0,
    help = 'Minimum accuracy of the model to be deployed'
)
def main(config: str, min_accuracy: float):
    """ Run the MLflow example pipeline."""
    # Get the MLflow model deployer stack component
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    deploy = config == DEPLOY or config == DEPLOY_AND_PREDICT
    predict = config == PREDICT or config == DEPLOY_AND_PREDICT

    if deploy:
        # Initialize a continuous deployment pipeline run
        continuous_deployment_pipeline(
            data_path = 'data/WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv',
            workers = 3
        )
    elif True:
        # Initialize a infrence pipeline run
        inference_pipeline(
            pipeline_name = 'continuous_deployment_pipeline',
            pipeline_step_name = 'mlflow_model_deployer_step'
        )

if __name__ == '__main__':
    main()