"""src/orchestration - Modules d'orchestration unifiée du pipeline Prefect 3."""

from src.orchestration.lifecycle import on_flow_completion, on_flow_failure
from src.orchestration.tasks import (
    steam_training_pipeline,
    task_evaluate_and_publish,
    task_load_data,
    task_preprocess_data,
    task_train_model,
)

__all__ = [
    "on_flow_completion",
    "on_flow_failure",
    "steam_training_pipeline",
    "task_load_data",
    "task_preprocess_data",
    "task_train_model",
    "task_evaluate_and_publish",
]
