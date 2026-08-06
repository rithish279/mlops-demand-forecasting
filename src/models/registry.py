import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "bike-demand-forecaster"

def register_latest_model(run_id: str, model_name: str = MODEL_NAME):
    model_uri = f"runs:/{run_id}/model"
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

    model, metrics = run_training()

    client = MlflowClient()
    latest_run = client.search_runs(
        experiment_ids=["0"], order_by=["start_time DESC"], max_results=1
    )[0]

    result = register_latest_model(latest_run.info.run_id)
    promote_to_staging(MODEL_NAME, result.version)
