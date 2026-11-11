"""src/tracking/mlflow_tracking.py - Suivi exhaustif des entraînements et des artefacts MLflow.

Fournit :
- Initialisation et configuration centralisée du client MLflow.
- Création / résolution d'expérience unifiée ("steam-mlops-experiments").
- Journalisation exhaustive des paramètres, métriques (avec baselines) et tags de reproductibilité.
- Sérialisation scikit-learn avec signatures d'inférence strictes sans variables cibles.
- Enregistrement des modèles dans le registre ("steam-positive-ratio", "steam-owner-range").
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Optional, Tuple
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.contracts import PreparedData
from src.settings import settings
from src.training import TrainedModelsBundle

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
logger = logging.getLogger(__name__)

DEFAULT_EXPERIMENT_NAME = "steam-mlops-experiments"
REGRESSION_MODEL_NAME = "steam-positive-ratio"
CLASSIFICATION_MODEL_NAME = "steam-owner-range"


def get_git_commit_sha() -> str:
    """Retourne l'empreinte SHA du commit Git courant, ou 'unknown' en cas d'erreur."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown"


def get_or_create_experiment(
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    client: Optional[MlflowClient] = None,
    tracking_uri: Optional[str] = None,
) -> str:
    """Crée ou récupère l'identifiant d'une expérience MLflow."""
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", settings.MLFLOW_TRACKING_URI)
    cli = client or MlflowClient(tracking_uri=uri)
    exp = cli.get_experiment_by_name(experiment_name)
    if exp is None:
        return cli.create_experiment(
            name=experiment_name,
            tags={
                "project": "steam-mlops-5mlde",
                "framework": "scikit-learn",
                "team": "5MLDE",
            },
        )
    return exp.experiment_id


def init_mlflow(
    tracking_uri: Optional[str] = None,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
) -> Tuple[MlflowClient, str]:
    """Configure l'URI de tracking et retourne le client MLflow ainsi que l'ID d'expérience."""
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", settings.MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    experiment_id = get_or_create_experiment(experiment_name=experiment_name, client=client, tracking_uri=uri)
    mlflow.set_experiment(experiment_name=experiment_name)
    return client, experiment_id


def log_training_run(
    models_bundle: TrainedModelsBundle,
    prepared_data: PreparedData,
    eval_metrics: Dict[str, Any],
    sample_df: pd.DataFrame,
    snapshot: Optional[SnapshotRef] = None,
    tracking_uri: Optional[str] = None,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    run_name: Optional[str] = None,
    register_models: bool = True,
    model_name_suffix: str = "",
) -> Dict[str, Any]:
    """Journalise une tentative d'entraînement complète dans MLflow avec paramètres, métriques et modèles sérialisés."""
    client, exp_id = init_mlflow(tracking_uri=tracking_uri, experiment_name=experiment_name)
    context = prepared_data.context
    snap = snapshot or getattr(prepared_data, "snapshot", None)

    reg_name = f"{REGRESSION_MODEL_NAME}{model_name_suffix}"
    cls_name = f"{CLASSIFICATION_MODEL_NAME}{model_name_suffix}"

    actual_run_name = run_name or f"train-{context.mode.lower()}-{context.run_id[:8]}"

    # Construction de l'exemple d'entrée strictement dénué de cibles ou fuites
    features = list(settings.ALLOWED_INFERENCE_FEATURES)
    clean_sample_df = sample_df[features].dropna().head(5).copy()

    tags: Dict[str, str] = {
        "run_id": context.run_id,
        "pipeline_mode": context.mode,
        "git_commit": get_git_commit_sha(),
        "python_version": sys.version.split()[0],
        "framework": "scikit-learn",
        "data_gx_validated": str(getattr(context, "gx_validated", getattr(context, "gx_success", True))),
    }
    if snap:
        tags["snapshot_path"] = str(snap.path)
        tags["snapshot_sha256"] = snap.sha256
        tags["snapshot_row_count"] = str(snap.row_count)

    with mlflow.start_run(
        experiment_id=exp_id,
        run_name=actual_run_name,
        tags=tags,
    ) as run:
        mlflow_run_id = run.info.run_id
        logger.info(f"[MLFLOW] Début du run MLflow : {mlflow_run_id} ({actual_run_name})")

        # ----------------------------------------------------------------------
        # 1. Paramètres
        # ----------------------------------------------------------------------
        params: Dict[str, Any] = {
            "random_seed": settings.RANDOM_SEED,
            "split.train_ratio": settings.TRAIN_RATIO,
            "split.val_ratio": settings.VAL_RATIO,
            "split.test_ratio": settings.TEST_RATIO,
            "features.top_genres_k": 15,
            "features.allowed_count": len(features),
            "features.allowed_names": ",".join(features),
            "regression.estimator": "RandomForestRegressor",
            "regression.n_estimators": 40,
            "regression.max_depth": 12,
            "regression.min_reviews": settings.REGRESSION_MIN_REVIEWS,
            "classification.estimator": "RandomForestClassifier",
            "classification.n_estimators": 40,
            "classification.max_depth": 12,
            "classification.min_class_count": settings.CLASSIFICATION_MIN_CLASS_COUNT,
            "classification.excluded_class": settings.CLASSIFICATION_EXCLUDED_CLASS,
        }
        mlflow.log_params(params)

        # ----------------------------------------------------------------------
        # 2. Métriques (Régression, Classification, Baselines et Durées)
        # ----------------------------------------------------------------------
        reg_eval = eval_metrics.get("regression", {})
        cls_eval = eval_metrics.get("classification", {})
        durations = eval_metrics.get("training_durations_sec", models_bundle.training_durations)

        metrics: Dict[str, float] = {
            # Durées
            "duration.reg_train_sec": float(durations.get("regression_seconds", 0.0)),
            "duration.cls_train_sec": float(durations.get("classification_seconds", 0.0)),
            "duration.total_train_sec": float(durations.get("total_training_seconds", 0.0)),
            # Régression Train / Val / Test
            "regression.train.mae": reg_eval["train"]["candidate"]["mae"],
            "regression.train.rmse": reg_eval["train"]["candidate"]["rmse"],
            "regression.train.r2": reg_eval["train"]["candidate"]["r2"],
            "regression.val.mae": reg_eval["validation"]["candidate"]["mae"],
            "regression.val.rmse": reg_eval["validation"]["candidate"]["rmse"],
            "regression.val.r2": reg_eval["validation"]["candidate"]["r2"],
            "regression.test.mae": reg_eval["test"]["candidate"]["mae"],
            "regression.test.rmse": reg_eval["test"]["candidate"]["rmse"],
            "regression.test.r2": reg_eval["test"]["candidate"]["r2"],
            # Régression Baselines
            "regression.val.baseline_mae": reg_eval["validation"]["baseline"]["mae"],
            "regression.val.baseline_rmse": reg_eval["validation"]["baseline"]["rmse"],
            "regression.val.baseline_r2": reg_eval["validation"]["baseline"]["r2"],
            # Classification Train / Val / Test
            "classification.train.accuracy": cls_eval["train"]["candidate"]["accuracy"],
            "classification.train.f1_macro": cls_eval["train"]["candidate"]["f1_macro"],
            "classification.val.accuracy": cls_eval["validation"]["candidate"]["accuracy"],
            "classification.val.f1_macro": cls_eval["validation"]["candidate"]["f1_macro"],
            "classification.test.accuracy": cls_eval["test"]["candidate"]["accuracy"],
            "classification.test.f1_macro": cls_eval["test"]["candidate"]["f1_macro"],
            # Classification Baselines
            "classification.val.baseline_accuracy": cls_eval["validation"]["baseline"]["accuracy"],
            "classification.val.baseline_f1_macro": cls_eval["validation"]["baseline"]["f1_macro"],
        }

        # Calcul d'amélioration relative vs baseline (validation)
        val_reg_cand_mae = reg_eval["validation"]["candidate"]["mae"]
        val_reg_base_mae = reg_eval["validation"]["baseline"]["mae"]
        if val_reg_base_mae > 0:
            metrics["regression.val.improvement_mae_pct"] = float(
                100.0 * (val_reg_base_mae - val_reg_cand_mae) / val_reg_base_mae
            )

        val_cls_cand_acc = cls_eval["validation"]["candidate"]["accuracy"]
        val_cls_base_acc = cls_eval["validation"]["baseline"]["accuracy"]
        if val_cls_base_acc > 0:
            metrics["classification.val.improvement_acc_pct"] = float(
                100.0 * (val_cls_cand_acc - val_cls_base_acc) / val_cls_base_acc
            )

        mlflow.log_metrics(metrics)

        # ----------------------------------------------------------------------
        # 3. Sérialisation des Modèles avec Signatures Strictes
        # ----------------------------------------------------------------------
        # a) Régression
        reg_pipe = models_bundle.regression_pipeline
        reg_pred_sample = reg_pipe.predict(clean_sample_df)
        reg_signature = infer_signature(clean_sample_df, reg_pred_sample)

        reg_model_info = mlflow.sklearn.log_model(
            sk_model=reg_pipe,
            artifact_path="model_regression",
            signature=reg_signature,
            input_example=clean_sample_df,
            registered_model_name=reg_name if register_models else None,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

        # b) Classification
        cls_pipe = models_bundle.classification_pipeline
        cls_pred_sample = cls_pipe.predict(clean_sample_df)
        cls_signature = infer_signature(clean_sample_df, cls_pred_sample)

        cls_model_info = mlflow.sklearn.log_model(
            sk_model=cls_pipe,
            artifact_path="model_classification",
            signature=cls_signature,
            input_example=clean_sample_df,
            registered_model_name=cls_name if register_models else None,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

        # ----------------------------------------------------------------------
        # 4. Journalisation des Artefacts Complémentaires
        # ----------------------------------------------------------------------
        # Dump temporaire du payload JSON complet des métriques
        metrics_tmp = Path("artifacts") / "mlflow_metrics_summary.json"
        metrics_tmp.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_tmp, "w", encoding="utf-8") as f:
            json.dump(eval_metrics, f, indent=2)
        mlflow.log_artifact(str(metrics_tmp), artifact_path="evaluation")

        logger.info(
            f"[MLFLOW] Run {mlflow_run_id} enregistré avec succès dans l'expérience {exp_id}."
        )

        return {
            "mlflow_run_id": mlflow_run_id,
            "experiment_id": exp_id,
            "tracking_uri": mlflow.get_tracking_uri(),
            "regression_model_uri": reg_model_info.model_uri,
            "classification_model_uri": cls_model_info.model_uri,
            "metrics": metrics,
        }
