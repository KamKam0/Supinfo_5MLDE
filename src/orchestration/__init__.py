"""src/orchestration - Modules d'orchestration unifiée du pipeline Prefect 3."""

from src.orchestration.alerts import (
    AlertDispatcher,
    AlertMessage,
    format_teams_card_payload,
    trigger_failure_alert,
)
from src.orchestration.cache_policy import (
    FeatureCache,
    build_features_pure,
    compute_code_hash,
    compute_feature_cache_key,
    task_build_features_pure,
)
from src.orchestration.deployment import (
    apply_deployment,
    get_deployment_definition,
    pause_deployment_schedule,
    resume_deployment_schedule,
    serve_deployment,
    trigger_deployment_run,
)
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
    "build_features_pure",
    "task_build_features_pure",
    "FeatureCache",
    "compute_feature_cache_key",
    "compute_code_hash",
    "AlertMessage",
    "AlertDispatcher",
    "format_teams_card_payload",
    "trigger_failure_alert",
    "get_deployment_definition",
    "apply_deployment",
    "pause_deployment_schedule",
    "resume_deployment_schedule",
    "serve_deployment",
    "trigger_deployment_run",
]
