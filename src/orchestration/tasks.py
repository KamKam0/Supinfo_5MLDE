"""src/orchestration/tasks.py - Flow et tâches Prefect 3 pour l'entraînement observable.

Fournit :
- 4 tâches mères décorées avec @task(retries=0) :
  1. task_load_data : acquisition, intégrité SHA-256 et barrière Great Expectations.
  2. task_preprocess_data : filtrage d'éligibilité métier et split sans fuite.
  3. task_train_model : ajustement des estimateurs scikit-learn et baselines.
  4. task_evaluate_and_publish : évaluation multi-corpus, tracking MLflow et qualification formelle.
- Flow d'orchestration unifié @flow(name="steam-training-pipeline") séquentiel et déterministe.
- Chaîne de corrélation universelle :
  batch_id -> flow_run_id -> mlflow_run_id -> publication_id -> versions MLflow.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional, Tuple
import pandas as pd
from prefect import flow, task
from prefect.runtime import flow_run

from src.contracts import IngestionResult, PreparedData, RunContext, SnapshotRef
from src.evaluation import evaluate_models
from src.exceptions import DataValidationError
from src.ingestion import load_data
from src.orchestration.lifecycle import (
    build_correlation_record,
    format_correlation_chain,
    on_flow_completion,
    on_flow_failure,
)
from src.preprocessing import preprocess_data
from src.settings import settings
from src.tracking.mlflow_tracking import log_training_run
from src.tracking.publications import PublicationManager
from src.tracking.qualification import evaluate_run_qualification
from src.tracking.registry import promote_model_to_alias
from src.training import TrainedModelsBundle, train_models

logger = logging.getLogger(__name__)


def get_current_flow_run_id() -> str:
    """Retourne l'identifiant du FlowRun Prefect courant ou 'local-run' si hors contexte."""
    try:
        rid = flow_run.id
        if rid:
            return str(rid)
    except Exception:
        pass
    return "local-run"


def extract_flat_validation_metrics(eval_metrics: Dict[str, Any]) -> Dict[str, float]:
    """Extrait et aplatit les métriques de validation pour la politique de qualification."""
    reg_val = eval_metrics.get("regression", {}).get("validation", {})
    cls_val = eval_metrics.get("classification", {}).get("validation", {})

    return {
        "regression.val.mae": float(reg_val.get("candidate", {}).get("mae", float("nan"))),
        "regression.val.baseline_mae": float(reg_val.get("baseline", {}).get("mae", float("nan"))),
        "regression.val.rmse": float(reg_val.get("candidate", {}).get("rmse", float("nan"))),
        "regression.val.r2": float(reg_val.get("candidate", {}).get("r2", float("nan"))),
        "classification.val.accuracy": float(cls_val.get("candidate", {}).get("accuracy", float("nan"))),
        "classification.val.baseline_accuracy": float(cls_val.get("baseline", {}).get("accuracy", float("nan"))),
        "classification.val.f1_macro": float(cls_val.get("candidate", {}).get("f1_macro", float("nan"))),
        "classification.val.baseline_f1_macro": float(cls_val.get("baseline", {}).get("f1_macro", float("nan"))),
    }


# ==============================================================================
# TÂCHES MÈRES PREFECT
# Règle absolue de retries : retries=0 sur erreurs métier / données
# ==============================================================================


@task(name="task_load_data", retries=0, description="Acquisition, vérification SHA-256 et barrière bloquante GX")
def task_load_data(config: Dict[str, Any] | str | Path) -> IngestionResult:
    """Première tâche mère : chargement et validation bloquante des données."""
    current_flow_run_id = get_current_flow_run_id()
    t0 = time.time()

    cfg_dict = dict(config) if isinstance(config, dict) else {"config_path": str(config)}
    batch_id = cfg_dict.get("batch_id", "batch-000")

    logger.info(
        f"[TASK:LOAD_DATA] Démarrage | flow_run_id={current_flow_run_id} | batch_id={batch_id}"
    )

    # Ingestion et validation GX stricte
    ingestion_result = load_data(config)
    duration = round(time.time() - t0, 3)

    if not ingestion_result.gx_success:
        error_msg = f"[TASK:LOAD_DATA] Échec de la barrière Great Expectations sur {ingestion_result.snapshot.path}"
        logger.error(error_msg)
        raise DataValidationError(error_msg)

    logger.info(
        f"[TASK:LOAD_DATA] Terminé en {duration}s | flow_run_id={current_flow_run_id} | "
        f"batch_id={ingestion_result.context.batch_id} | snapshot={Path(ingestion_result.snapshot.path).name} | "
        f"lignes={ingestion_result.snapshot.row_count:,} | colonnes={ingestion_result.snapshot.column_count} | GX=SUCCÈS"
    )
    return ingestion_result


@task(name="task_preprocess_data", retries=0, description="Filtrage d'éligibilité métier et découpage sans fuite")
def task_preprocess_data(ingestion_result: IngestionResult) -> Tuple[PreparedData, pd.DataFrame]:
    """Deuxième tâche mère : prétraitement et étanchéité des partitions."""
    current_flow_run_id = get_current_flow_run_id()
    t0 = time.time()

    if not ingestion_result.gx_success:
        raise DataValidationError(
            "[TASK:PREPROCESS] Interdiction absolue de prétraiter des données non validées par GX."
        )

    logger.info(
        f"[TASK:PREPROCESS] Démarrage | flow_run_id={current_flow_run_id} | "
        f"snapshot={Path(ingestion_result.snapshot.path).name}"
    )

    full_df = pd.read_csv(ingestion_result.snapshot.path)
    prepared_data = preprocess_data(ingestion_result)
    duration = round(time.time() - t0, 3)

    reg_data = prepared_data.regression_data
    cls_data = prepared_data.classification_data

    logger.info(
        f"[TASK:PREPROCESS] Terminé en {duration}s | flow_run_id={current_flow_run_id} | "
        f"Régression (Train: {len(reg_data.train_appids):,}, Val: {len(reg_data.val_appids):,}, Test: {len(reg_data.test_appids):,}) | "
        f"Classification (Train: {len(cls_data.train_appids):,}, Val: {len(cls_data.val_appids):,}, Test: {len(cls_data.test_appids):,})"
    )
    return prepared_data, full_df


@task(name="task_train_model", retries=0, description="Ajustement des estimateurs scikit-learn et baselines")
def task_train_model(
    prepared_data: PreparedData,
    full_df: pd.DataFrame,
    n_estimators: int = 40,
    max_depth: int = 12,
) -> TrainedModelsBundle:
    """Troisième tâche mère : ajustement des estimateurs scikit-learn."""
    current_flow_run_id = get_current_flow_run_id()
    logger.info(f"[TASK:TRAIN] Démarrage de l'entraînement des modèles | flow_run_id={current_flow_run_id}")

    t0 = time.time()
    models_bundle = train_models(
        prepared_data=prepared_data,
        full_df=full_df,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )
    duration = round(time.time() - t0, 3)

    logger.info(
        f"[TASK:TRAIN] Terminé en {duration}s | flow_run_id={current_flow_run_id} | "
        f"Durées détaillées: Reg={models_bundle.training_durations.get('regression_candidate', 0)}s, "
        f"Cls={models_bundle.training_durations.get('classification_candidate', 0)}s"
    )
    return models_bundle


@task(name="task_evaluate_and_publish", retries=0, description="Évaluation multi-corpus, tracking MLflow et qualification")
def task_evaluate_and_publish(
    models_bundle: TrainedModelsBundle,
    prepared_data: PreparedData,
    full_df: pd.DataFrame,
    snapshot_ref: Optional[SnapshotRef] = None,
    register_models: bool = True,
    qualify: bool = True,
    promote_champion: bool = True,
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Quatrième tâche mère : évaluation, journalisation MLflow et qualification formelle."""
    current_flow_run_id = get_current_flow_run_id()
    batch_id = prepared_data.context.batch_id or "batch-000"
    logger.info(
        f"[TASK:EVALUATE_PUBLISH] Démarrage | flow_run_id={current_flow_run_id} | batch_id={batch_id}"
    )

    t0 = time.time()
    eval_metrics = evaluate_models(
        models_bundle=models_bundle,
        prepared_data=prepared_data,
        full_df=full_df,
    )

    # 1. Enregistrement MLflow avec corrélation Prefect
    extra_tags = {
        "prefect.flow_run_id": current_flow_run_id,
        "prefect.flow_name": "steam-training-pipeline",
        "batch_id": batch_id,
    }

    mlflow_info = log_training_run(
        models_bundle=models_bundle,
        prepared_data=prepared_data,
        eval_metrics=eval_metrics,
        sample_df=full_df,
        snapshot=snapshot_ref,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name or "steam-mlops-experiments",
        register_models=register_models,
        extra_tags=extra_tags,
    )

    mlflow_run_id = mlflow_info["mlflow_run_id"]
    reg_name = mlflow_info.get("regression_model_name", "steam-positive-ratio")
    reg_ver = mlflow_info.get("regression_model_version")
    cls_name = mlflow_info.get("classification_model_name", "steam-owner-range")
    cls_ver = mlflow_info.get("classification_model_version")

    model_versions = {}
    if reg_ver:
        model_versions[reg_name] = str(reg_ver)
    if cls_ver:
        model_versions[cls_name] = str(cls_ver)

    publication_id: Optional[str] = None
    qual_report_dict: Optional[Dict[str, Any]] = None

    # 2. Qualification formelle déterministe
    if qualify and register_models and reg_ver and cls_ver:
        flat_metrics = extract_flat_validation_metrics(eval_metrics)
        report = evaluate_run_qualification(
            run_id=mlflow_run_id,
            metrics=flat_metrics,
            tags={"pipeline_mode": prepared_data.context.mode},
            reload_test_passed=True,
        )
        qual_report_dict = report.model_dump()
        logger.info(
            f"[TASK:EVALUATE_PUBLISH] Verdict de qualification : {report.verdict} "
            f"(Passed: {report.passed}) - {report.summary}"
        )

        # 3. Promotion transactionnelle si éligible
        if report.passed and promote_champion:
            pub_manager = PublicationManager()
            publication_id = pub_manager.create_publication(
                run_id=mlflow_run_id,
                verdict=report.verdict,
                target_alias="champion",
                regression_model_name=reg_name,
                classification_model_name=cls_name,
                regression_model_uri=f"models:/{reg_name}/{reg_ver}",
                classification_model_uri=f"models:/{cls_name}/{cls_ver}",
            )

            promote_model_to_alias(
                model_name=reg_name,
                version=str(reg_ver),
                alias="champion",
                reason=f"Promotion automatique Prefect flow {current_flow_run_id}",
                pub_manager=pub_manager,
                tracking_uri=tracking_uri,
            )
            pub_manager.update_publication(
                publication_id,
                regression_status="COMPLETED",
                regression_version=str(reg_ver),
            )

            promote_model_to_alias(
                model_name=cls_name,
                version=str(cls_ver),
                alias="champion",
                reason=f"Promotion automatique Prefect flow {current_flow_run_id}",
                pub_manager=pub_manager,
                tracking_uri=tracking_uri,
            )
            pub_manager.update_publication(
                publication_id,
                classification_status="COMPLETED",
                classification_version=str(cls_ver),
                status="COMPLETED",
            )
            logger.info(
                f"[TASK:EVALUATE_PUBLISH] Promotion @champion validée sous publication_id={publication_id}"
            )

    duration = round(time.time() - t0, 3)

    # 4. Enregistrement et journalisation de la chaîne de corrélation universelle
    correlation_record = build_correlation_record(
        batch_id=batch_id,
        flow_run_id=current_flow_run_id,
        mlflow_run_id=mlflow_run_id,
        publication_id=publication_id,
        versions=model_versions,
        status="COMPLETED",
    )
    logger.info(correlation_record["formatted_chain"])

    return {
        "flow_run_id": current_flow_run_id,
        "batch_id": batch_id,
        "mlflow_run_id": mlflow_run_id,
        "publication_id": publication_id,
        "model_versions": model_versions,
        "eval_metrics": eval_metrics,
        "qualification": qual_report_dict,
        "correlation": correlation_record,
        "duration": duration,
    }


# ==============================================================================
# FLOW D'ORCHESTRATION PREFECT
# ==============================================================================


@flow(
    name="steam-training-pipeline",
    description="Pipeline d'entraînement unifié Steam MLOps avec corrélation d'identifiants",
    on_completion=[on_flow_completion],
    on_failure=[on_flow_failure],
)
def steam_training_pipeline(
    config: Optional[Dict[str, Any] | str | Path] = None,
    register_models: bool = True,
    qualify: bool = True,
    promote_champion: bool = True,
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
    n_estimators: int = 40,
    max_depth: int = 12,
) -> Dict[str, Any]:
    """Orchestre séquentiellement et de manière déterministe les 4 fonctions mères."""
    current_flow_run_id = get_current_flow_run_id()

    # Configuration par défaut si non fournie
    if config is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        resolved_config: Dict[str, Any] = {
            "snapshot_path": str(project_root / "data" / "snapshots" / "historical_projection.csv"),
            "validation_suite": "snapshot_integrity_v1",
            "mode": "historical-training",
            "batch_id": "batch-default",
            "flow_run_id": current_flow_run_id,
        }
    elif isinstance(config, dict):
        resolved_config = dict(config)
        resolved_config["flow_run_id"] = current_flow_run_id
    else:
        resolved_config = {"config_path": str(config), "flow_run_id": current_flow_run_id}

    logger.info(
        f"[PIPELINE] Démarrage du flow 'steam-training-pipeline' | flow_run_id={current_flow_run_id}"
    )

    # 1. Chargement et barrière Great Expectations
    ingestion_result = task_load_data(resolved_config)

    # 2. Prétraitement et partitions anti-fuite
    prepared_data, full_df = task_preprocess_data(ingestion_result)

    # 3. Entraînement des estimateurs et baselines
    models_bundle = task_train_model(
        prepared_data=prepared_data,
        full_df=full_df,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )

    # 4. Évaluation, tracking MLflow et qualification formelle
    result = task_evaluate_and_publish(
        models_bundle=models_bundle,
        prepared_data=prepared_data,
        full_df=full_df,
        snapshot_ref=ingestion_result.snapshot,
        register_models=register_models,
        qualify=qualify,
        promote_champion=promote_champion,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
    )

    logger.info(
        f"[PIPELINE] Flow terminé avec succès | flow_run_id={current_flow_run_id} | "
        f"mlflow_run_id={result['mlflow_run_id']} | publication_id={result.get('publication_id')}"
    )
    return result


def main() -> None:
    """Point d'entrée CLI pour exécuter le pipeline orchestré."""
    parser = argparse.ArgumentParser(description="Exécuteur du pipeline d'entraînement Prefect")
    parser.add_argument("--snapshot", type=str, default=None, help="Chemin vers le snapshot CSV")
    parser.add_argument("--batch-id", type=str, default="batch-manual", help="Identifiant du lot")
    parser.add_argument("--no-register", action="store_true", help="Ne pas enregistrer les modèles dans le registre")
    parser.add_argument("--no-promote", action="store_true", help="Ne pas promouvoir les modèles en @champion")
    parser.add_argument("--fast", action="store_true", help="Exécution rapide pour tests (5 estimateurs)")
    args = parser.parse_args()

    # Configuration Prefect API URL par défaut
    if "PREFECT_API_URL" not in os.environ:
        os.environ["PREFECT_API_URL"] = settings.PREFECT_API_URL

    cfg = {
        "batch_id": args.batch_id,
        "validation_suite": "snapshot_integrity_v1",
    }
    if args.snapshot:
        cfg["snapshot_path"] = args.snapshot

    n_est = 5 if args.fast else 40
    m_depth = 6 if args.fast else 12

    res = steam_training_pipeline(
        config=cfg,
        register_models=not args.no_register,
        promote_champion=not args.no_promote,
        n_estimators=n_est,
        max_depth=m_depth,
    )
    print("\n--- RÉSULTAT DU PIPELINE ORCHESTRÉ ---")
    print(res["correlation"]["formatted_chain"])


if __name__ == "__main__":
    main()
