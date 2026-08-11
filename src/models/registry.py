import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient

MODEL_NAME = "bike-demand-forecaster"


def register_model_from_uri(model_uri: str, model_name: str = MODEL_NAME):
    """Registers a model using its exact MLflow-provided URI."""
    result = mlflow.register_model(model_uri, model_name)
    print(f"Registered {model_name} version {result.version}")
    return result


def promote_to_staging(model_name: str, version: str):
    client = MlflowClient()
    client.set_registered_model_alias(model_name, "staging", version)
    print(f"{model_name} v{version} promoted to Staging")


def promote_to_production(model_name: str, version: str):
    client = MlflowClient()
    client.set_registered_model_alias(model_name, "production", version)
    print(f"{model_name} v{version} promoted to Production")


def get_production_model(model_name: str = MODEL_NAME):
    model_uri = f"models:/{model_name}@production"
    return mlflow.xgboost.load_model(model_uri)


if __name__ == "__main__":
    from src.models.train import run_training

    model, metrics, model_info = run_training()

    result = register_model_from_uri(model_info.model_uri)
    promote_to_staging(MODEL_NAME, result.version)