"""src/tracking/reconciliation.py - Moteur de réconciliation et reprise après interruption.

Garantit :
- L'idempotence stricte des opérations de publication du Model Registry.
- La reprise déterministe d'un jalon interrompu (crash, perte réseau, arrêt conteneur).
- L'interdiction absolue de créer des versions dupliquées ou orphelines dans MLflow.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional
import mlflow
from mlflow.tracking import MlflowClient

from src.settings import settings
from src.tracking.publications import PublicationManager
from src.tracking.registry import get_client, set_model_alias

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
logger = logging.getLogger(__name__)


def find_existing_model_version_for_run(
    client: MlflowClient,
    model_name: str,
    run_id: str,
) -> Optional[str]:
    """Recherche dans MLflow si une version est déjà enregistrée pour ce nom et ce run_id."""
    try:
        versions = client.search_model_versions(f"name='{model_name}' and run_id='{run_id}'")
        if versions:
            # S'il y a déjà une version, on prend la plus récente pour ce run
            sorted_v = sorted(versions, key=lambda v: int(v.version), reverse=True)
            return str(sorted_v[0].version)
    except Exception as exc:
        logger.warning(f"[RECONCILIATION] Erreur recherche version existante pour {model_name} : {exc}")
    return None


def reconcile_publication(
    publication_id: str,
    client: Optional[MlflowClient] = None,
    pub_manager: Optional[PublicationManager] = None,
    tracking_uri: Optional[str] = None,
) -> Dict[str, Any]:
    """Reprend et réconcilie une publication interrompue de façon 100 % idempotente."""
    manager = pub_manager or PublicationManager()
    cli = client or get_client(tracking_uri)

    pub = manager.get_publication(publication_id)
    if not pub:
        raise ValueError(f"Publication '{publication_id}' introuvable dans le journal SQLite.")

    if pub["status"] == "COMPLETED":
        logger.info(f"[RECONCILIATION] Publication {publication_id} déjà COMPLETED. Aucune action requise.")
        return {
            "publication_id": publication_id,
            "status": "COMPLETED",
            "already_completed": True,
            "regression_version": pub["regression_version"],
            "classification_version": pub["classification_version"],
            "reconciled_steps": [],
        }

    logger.info(f"[RECONCILIATION] Début de réconciliation pour {publication_id} (état: {pub['status']})...")
    reconciled_steps: List[str] = []
    run_id = pub["run_id"]
    target_alias = pub["target_alias"].lstrip("@")
    reg_name = pub["regression_model_name"] or "steam-positive-ratio"
    cls_name = pub["classification_model_name"] or "steam-owner-range"
    reg_uri = pub["regression_model_uri"] or f"runs:/{run_id}/model_regression"
    cls_uri = pub["classification_model_uri"] or f"runs:/{run_id}/model_classification"

    manager.update_publication(publication_id, status="IN_PROGRESS")

    try:
        # ----------------------------------------------------------------------
        # 1. Réconciliation Modèle Régression
        # ----------------------------------------------------------------------
        reg_ver = pub["regression_version"]
        if not reg_ver:
            # Vérifier si la version a déjà été créée dans MLflow avant l'interruption
            existing_reg_ver = find_existing_model_version_for_run(cli, reg_name, run_id)
            if existing_reg_ver:
                reg_ver = existing_reg_ver
                reconciled_steps.append(f"regression_version_recovered_v{reg_ver}")
            else:
                # Création propre dans MLflow
                mv = cli.create_model_version(name=reg_name, source=reg_uri, run_id=run_id)
                reg_ver = str(mv.version)
                reconciled_steps.append(f"regression_version_created_v{reg_ver}")
            manager.update_publication(publication_id, regression_version=reg_ver, regression_status="REGISTERED")

        if pub["regression_status"] != "ALIASED":
            set_model_alias(reg_name, target_alias, reg_ver, client=cli)
            manager.update_publication(publication_id, regression_status="ALIASED")
            reconciled_steps.append(f"regression_alias_{target_alias}_assigned")

        # ----------------------------------------------------------------------
        # 2. Réconciliation Modèle Classification
        # ----------------------------------------------------------------------
        cls_ver = pub["classification_version"]
        if not cls_ver:
            existing_cls_ver = find_existing_model_version_for_run(cli, cls_name, run_id)
            if existing_cls_ver:
                cls_ver = existing_cls_ver
                reconciled_steps.append(f"classification_version_recovered_v{cls_ver}")
            else:
                mv = cli.create_model_version(name=cls_name, source=cls_uri, run_id=run_id)
                cls_ver = str(mv.version)
                reconciled_steps.append(f"classification_version_created_v{cls_ver}")
            manager.update_publication(publication_id, classification_version=cls_ver, classification_status="REGISTERED")

        if pub["classification_status"] != "ALIASED":
            set_model_alias(cls_name, target_alias, cls_ver, client=cli)
            manager.update_publication(publication_id, classification_status="ALIASED")
            reconciled_steps.append(f"classification_alias_{target_alias}_assigned")

        # ----------------------------------------------------------------------
        # 3. Finalisation et Journalisation d'Événements
        # ----------------------------------------------------------------------
        manager.record_selection_event(
            model_name=reg_name,
            target_version=reg_ver,
            alias=target_alias,
            action="PROMOTION",
            reason=f"Publication réconciliée ({publication_id})",
            publication_id=publication_id,
        )
        manager.record_selection_event(
            model_name=cls_name,
            target_version=cls_ver,
            alias=target_alias,
            action="PROMOTION",
            reason=f"Publication réconciliée ({publication_id})",
            publication_id=publication_id,
        )

        manager.update_publication(publication_id, status="COMPLETED")
        logger.info(f"[RECONCILIATION] Publication {publication_id} réconciliée avec SUCCÈS.")

        return {
            "publication_id": publication_id,
            "status": "COMPLETED",
            "already_completed": False,
            "regression_version": reg_ver,
            "classification_version": cls_ver,
            "reconciled_steps": reconciled_steps,
        }

    except Exception as exc:
        err_msg = f"Échec de réconciliation : {exc}"
        logger.error(f"[RECONCILIATION ERROR] {err_msg}")
        manager.update_publication(publication_id, status="FAILED", error_message=err_msg)
        raise
