from azure.ai.ml import MLClient
from azure.ai.ml.entities import CommandJob, Environment
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id="TON_SUB_ID",
    resource_group_name="TON_RG",
    workspace_name="TON_WORKSPACE"
)

env = Environment(
    name="real-estate-env",
    description="Environment with Papermill for notebook execution",
    conda_file="environments/azure_env.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04"
)

job = CommandJob(
    code="./",  # Tout le repo sera envoyé
    command="python scripts/pipeline_runner.py",
    environment=env,
    compute="cpu-cluster",  # Nom exact de ton cluster dans Azure ML
    display_name="run-notebooks-pipeline",
    experiment_name="real-estate-notebook-pipeline",
)

ml_client.jobs.create_or_update(job)
