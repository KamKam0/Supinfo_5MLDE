"""src/tracking/registry.py - Gestion du Model Registry et Rechargement Externe.

Fournit :
- Fonctions de rechargement sécurisé de modèles scikit-learn depuis MLflow (`load_model`).
- Résolution des URIs du Model Registry (`models:/<model_name>/<version>`).
- Validation des entrées d'inférence (contrôle des colonnes autorisées et exclusion des fuites).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.contracts import InferenceInputGame
from src.settings import settings

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
logger = logging.getLogger(__name__)


def get_client(tracking_uri: Optional[str] = None) -> MlflowClient:
    """Retourne une instance configurée de MlflowClient."""
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", settings.MLFLOW_TRACKING_URI)
    return MlflowClient(tracking_uri=uri)


def load_model(model_uri: str, tracking_uri: Optional[str] = None) -> Pipeline:
    """Charge un pipeline scikit-learn sérialisé depuis MLflow (runs:/ ou models:/)."""
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", settings.MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(uri)
    logger.info(f"[MLFLOW REGISTRY] Rechargement du modèle depuis : {model_uri}")
    model = mlflow.sklearn.load_model(model_uri=model_uri)
    if not isinstance(model, Pipeline):
        logger.warning(f"Le modèle rechargé ({type(model)}) n'est pas une instance explicite de sklearn.pipeline.Pipeline.")
    return model


def get_latest_model_uri(model_name: str, tracking_uri: Optional[str] = None) -> str:
    """Retourne l'URI de la dernière version enregistrée d'un modèle dans le registre MLflow."""
    client = get_client(tracking_uri)
    try:
        versions = client.get_latest_versions(name=model_name)
        if not versions:
            raise ValueError(f"Aucune version enregistrée trouvée pour le modèle '{model_name}'.")
        # Sélection de la version la plus récente
        latest_version = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
        return f"models:/{model_name}/{latest_version.version}"
    except Exception as exc:
        logger.error(f"[MLFLOW REGISTRY] Erreur lors de la résolution de '{model_name}' : {exc}")
        raise


def get_registered_model_details(model_name: str, tracking_uri: Optional[str] = None) -> Dict[str, Any]:
    """Retourne les métadonnées et l'historique des versions d'un modèle enregistré."""
    client = get_client(tracking_uri)
    model = client.get_registered_model(model_name)
    versions = client.search_model_versions(f"name='{model_name}'")

    return {
        "name": model.name,
        "description": model.description,
        "creation_timestamp": model.creation_timestamp,
        "last_updated_timestamp": model.last_updated_timestamp,
        "versions": [
            {
                "version": v.version,
                "current_stage": v.current_stage,
                "run_id": v.run_id,
                "status": v.status,
                "source": v.source,
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

    # Vérification anti-fuite : aucune cible ou variable fuyante ne doit être transmise
    forbidden_present = set(settings.LEAKAGE_COLUMNS).intersection(set(df.columns))
    if forbidden_present:
        raise ValueError(
            f"Violation d'étanchéité : présence de colonnes réservées/fuyantes lors de l'inférence : {forbidden_present}"
        )

    # Vérification que toutes les colonnes requises sont fournies
    missing_cols = set(cols_to_use) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Colonnes obligatoires manquantes pour l'inférence : {missing_cols}")

    X = df[cols_to_use].copy()
    return model.predict(X)
