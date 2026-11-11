"""src/tracking - Module d'observabilité, de tracking et de gouvernance des modèles (MLflow)."""

from src.tracking.mlflow_tracking import (
    get_or_create_experiment,
    init_mlflow,
    log_training_run,
)
from src.tracking.publications import (
    PublicationManager,
)
from src.tracking.qualification import (
    QualificationReport,
    evaluate_run_qualification,
    load_qualification_policy,
)
from src.tracking.reconciliation import (
    reconcile_publication,
)
from src.tracking.registry import (
    get_latest_model_uri,
    get_model_version_by_alias,
    get_registered_model_details,
    load_model,
    predict_pipeline,
    promote_model_to_alias,
    rollback_model_alias,
    set_model_alias,
)

__all__ = [
    "init_mlflow",
    "get_or_create_experiment",
    "log_training_run",
    "load_model",
    "get_latest_model_uri",
    "get_registered_model_details",
    "predict_pipeline",
    "set_model_alias",
    "get_model_version_by_alias",
    "promote_model_to_alias",
    "rollback_model_alias",
    "evaluate_run_qualification",
    "load_qualification_policy",
    "QualificationReport",
    "PublicationManager",
    "reconcile_publication",
]
