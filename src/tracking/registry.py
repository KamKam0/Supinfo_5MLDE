"""src/tracking/registry.py - Gestion du Model Registry, Alias et Rechargement Externe.

Fournit :
- Fonctions de rechargement sécurisé de modèles scikit-learn depuis MLflow (`load_model`).
- Gouvernance des alias officiels du registre (`@champion`, `@challenger`, `@candidate`).
- Promotion atomique de versions de modèles avec horodatage et audit.
- Procédure de retour arrière (*rollback*) vers une version antérieure saine.
- Validation des entrées d'inférence (contrôle des colonnes autorisées et exclusion des fuites).
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
from typing import Any, Dict, List, Optional
import mlflow
import mlflow.exceptions
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.contracts import InferenceInputGame
from src.settings import settings
from src.tracking.publications import PublicationManager

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
logger = logging.getLogger(__name__)


def get_client(tracking_uri: Optional[str] = None) -> MlflowClient:
    """Retourne une instance configurée de MlflowClient."""
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", settings.MLFLOW_TRACKING_URI)
    return MlflowClient(tracking_uri=uri)


def load_model(model_uri: str, tracking_uri: Optional[str] = None) -> Pipeline:
    """Charge un pipeline scikit-learn sérialisé depuis MLflow (runs:/, models:/, ou alias)."""
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", settings.MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(uri)
    logger.info(f"[MLFLOW REGISTRY] Rechargement du modèle depuis : {model_uri}")
    model = mlflow.sklearn.load_model(model_uri=model_uri)
    if not isinstance(model, Pipeline):
        logger.warning(
            f"Le modèle rechargé ({type(model)}) n'est pas une instance explicite de sklearn.pipeline.Pipeline."
        )
    return model


def get_latest_model_uri(model_name: str, tracking_uri: Optional[str] = None) -> str:
    """Retourne l'URI de la dernière version enregistrée d'un modèle dans le registre MLflow."""
    client = get_client(tracking_uri)
    try:
        versions = client.get_latest_versions(name=model_name)
        if not versions:
            raise ValueError(f"Aucune version enregistrée trouvée pour le modèle '{model_name}'.")
        latest_version = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
        return f"models:/{model_name}/{latest_version.version}"
    except Exception as exc:
        logger.error(f"[MLFLOW REGISTRY] Erreur lors de la résolution de '{model_name}' : {exc}")
        raise


def set_model_alias(
    model_name: str,
    alias: str,
    version: str,
    client: Optional[MlflowClient] = None,
    tracking_uri: Optional[str] = None,
) -> None:
    """Assigne un alias formel (ex. 'champion', 'challenger') à une version de modèle."""
    cli = client or get_client(tracking_uri)
    clean_alias = alias.lstrip("@")
    cli.set_registered_model_alias(name=model_name, alias=clean_alias, version=str(version))
    logger.info(f"[MLFLOW REGISTRY] Alias '@{clean_alias}' assigné à {model_name} (v{version}).")


def get_model_version_by_alias(
    model_name: str,
    alias: str,
    client: Optional[MlflowClient] = None,
    tracking_uri: Optional[str] = None,
) -> Optional[str]:
    """Retourne le numéro de version actuellement associé à un alias, ou None si inexistant."""
    cli = client or get_client(tracking_uri)
    clean_alias = alias.lstrip("@")
    try:
        mv = cli.get_model_version_by_alias(name=model_name, alias=clean_alias)
        return str(mv.version)
    except mlflow.exceptions.RestException as exc:
        if "RESOURCE_DOES_NOT_EXIST" in str(exc) or "not found" in str(exc).lower():
            return None
        raise


def promote_model_to_alias(
    model_name: str,
    version: str,
    alias: str = "champion",
    reason: str = "Promotion après qualification formelle",
    triggered_by: str = "coordinator",
    qualification_status: str = "production-ready",
    pub_manager: Optional[PublicationManager] = None,
    client: Optional[MlflowClient] = None,
    tracking_uri: Optional[str] = None,
) -> Dict[str, Any]:
    """Promeut une version de modèle vers un alias cible et consigne l'événement d'audit."""
    cli = client or get_client(tracking_uri)
    clean_alias = alias.lstrip("@")
    prev_version = get_model_version_by_alias(model_name, clean_alias, client=cli)

    # 1. Mise à jour des tags de gouvernance sur la version dans MLflow
    now_iso = datetime.now(timezone.utc).isoformat()
    cli.set_model_version_tag(model_name, str(version), "qualification", qualification_status)
    cli.set_model_version_tag(model_name, str(version), "promoted_at", now_iso)
    cli.set_model_version_tag(model_name, str(version), "promoted_by", triggered_by)

    # 2. Attribution de l'alias
    set_model_alias(model_name, clean_alias, str(version), client=cli)

    # 3. Journalisation de l'événement dans SQLite
    manager = pub_manager or PublicationManager()
    evt_id = manager.record_selection_event(
        model_name=model_name,
        target_version=str(version),
        alias=clean_alias,
        previous_version=prev_version,
        action="PROMOTION",
        reason=reason,
        triggered_by=triggered_by,
    )

    return {
        "event_id": evt_id,
        "model_name": model_name,
        "new_version": str(version),
        "previous_version": prev_version,
        "alias": clean_alias,
        "action": "PROMOTION",
        "timestamp": now_iso,
    }


def rollback_model_alias(
    model_name: str,
    target_version: str,
    alias: str = "champion",
    reason: str = "Retour arrière d'urgence sur anomalie",
    triggered_by: str = "operator",
    pub_manager: Optional[PublicationManager] = None,
    client: Optional[MlflowClient] = None,
    tracking_uri: Optional[str] = None,
) -> Dict[str, Any]:
    """Réactive une version antérieure saine en tant que champion et consigne l'incident."""
    cli = client or get_client(tracking_uri)
    clean_alias = alias.lstrip("@")
    prev_version = get_model_version_by_alias(model_name, clean_alias, client=cli)

    if prev_version == str(target_version):
        logger.warning(
            f"[ROLLBACK] La version {target_version} est déjà la version active pour l'alias '@{clean_alias}'."
        )

    now_iso = datetime.now(timezone.utc).isoformat()
    cli.set_model_version_tag(model_name, str(target_version), "rollback_at", now_iso)
    cli.set_model_version_tag(model_name, str(target_version), "rollback_reason", reason)

    # Réattribution de l'alias vers la version antérieure
    set_model_alias(model_name, clean_alias, str(target_version), client=cli)

    # Journalisation de l'audit de rollback
    manager = pub_manager or PublicationManager()
    evt_id = manager.record_selection_event(
        model_name=model_name,
        target_version=str(target_version),
        alias=clean_alias,
        previous_version=prev_version,
        action="ROLLBACK",
        reason=reason,
        triggered_by=triggered_by,
    )

    return {
        "event_id": evt_id,
        "model_name": model_name,
        "restored_version": str(target_version),
        "replaced_version": prev_version,
        "alias": clean_alias,
        "action": "ROLLBACK",
        "reason": reason,
        "timestamp": now_iso,
    }


def get_registered_model_details(
    model_name: str,
    tracking_uri: Optional[str] = None,
) -> Dict[str, Any]:
    """Retourne les métadonnées complètes, les alias et l'historique des versions."""
    client = get_client(tracking_uri)
    model = client.get_registered_model(model_name)
    versions = client.search_model_versions(f"name='{model_name}'")

    aliases = dict(getattr(model, "aliases", {}))

    return {
        "name": model.name,
        "description": model.description,
        "creation_timestamp": model.creation_timestamp,
        "last_updated_timestamp": model.last_updated_timestamp,
        "aliases": aliases,
        "versions": [
            {
                "version": v.version,
                "current_stage": v.current_stage,
                "run_id": v.run_id,
                "status": v.status,
                "source": v.source,
                "tags": dict(v.tags) if v.tags else {},
            }
            for v in versions
        ],
    }


def predict_pipeline(
    model: Pipeline,
    df: pd.DataFrame,
    features_subset: Optional[List[str]] = None,
) -> np.ndarray:
    """Exécute l'inférence en garantissant l'étanchéité et l'absence de colonnes cibles ou de fuite."""
    cols_to_use = features_subset or list(settings.ALLOWED_INFERENCE_FEATURES)

    forbidden_present = set(settings.LEAKAGE_COLUMNS).intersection(set(df.columns))
    if forbidden_present:
        raise ValueError(
            f"Violation d'étanchéité : présence de colonnes réservées/fuyantes lors de l'inférence : {forbidden_present}"
        )

    missing_cols = set(cols_to_use) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Colonnes obligatoires manquantes pour l'inférence : {missing_cols}")

    X = df[cols_to_use].copy()
    return model.predict(X)
