from mlflow.tracking import MlflowClient


def get_mlflow_client():
    """
    Initializes and returns an MLflow client instance.

    Returns:
        MlflowClient: An instance of the MLflow client.
    """
    return MlflowClient()
