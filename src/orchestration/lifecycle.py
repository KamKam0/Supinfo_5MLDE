"""src/orchestration/lifecycle.py - Cycle de vie des flows Prefect et corrélation universelle.

Fournit :
- Les hooks d'état Prefect 3 (on_flow_completion, on_flow_failure) garantissant la finalisation
  propre et déterministe des runs MLflow sans aucun run orphelin.
- Le formateur et constructeur de la chaîne de corrélation universelle des identifiants :
  batch_id -> flow_run_id -> mlflow_run_id -> publication_id -> model_versions.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any, Dict, Optional
import mlflow

logger = logging.getLogger(__name__)


def on_flow_completion(flow: Any, flow_run: Any, state: Any) -> None:
    """Hook Prefect exécuté lors de la complétion réussie d'un flow.

    Garantit la clôture propre de tout run MLflow actif restant ouvert.
    """
    flow_name = getattr(flow, "name", "unknown_flow")
    flow_run_id = str(getattr(flow_run, "id", "unknown_run_id"))
    logger.info(
        f"[PREFECT LIFECYCLE] Flow '{flow_name}' ({flow_run_id}) achevé avec succès (state: {state.name})."
    )

    active_run = mlflow.active_run()
    if active_run:
        active_id = active_run.info.run_id
        logger.info(
            f"[PREFECT LIFECYCLE] Clôture du run MLflow actif {active_id} en statut 'FINISHED'."
        )
        try:
            mlflow.end_run(status="FINISHED")
        except Exception as exc:
            logger.warning(f"[PREFECT LIFECYCLE] Erreur lors de la clôture MLflow : {exc}")


def on_flow_failure(flow: Any, flow_run: Any, state: Any) -> None:
    """Hook Prefect exécuté lors de l'échec d'un flow.

    Garantit le marquage en 'FAILED' de tout run MLflow orphelin sans persistance incohérente.
    """
    flow_name = getattr(flow, "name", "unknown_flow")
    flow_run_id = str(getattr(flow_run, "id", "unknown_run_id"))
    state_msg = getattr(state, "message", "Erreur indéterminée")
    logger.error(
        f"[PREFECT LIFECYCLE] Échec du Flow '{flow_name}' ({flow_run_id}) : {state_msg}"
    )

    active_run = mlflow.active_run()
    if active_run:
        active_id = active_run.info.run_id
        logger.warning(
            f"[PREFECT LIFECYCLE] Interruption prématurée détectée - Clôture immédiate du run MLflow "
            f"{active_id} en statut 'FAILED'."
        )
        try:
            mlflow.end_run(status="FAILED")
        except Exception as exc:
            logger.warning(f"[PREFECT LIFECYCLE] Erreur lors du marquage FAILED du run MLflow : {exc}")


def format_correlation_chain(
    batch_id: str,
    flow_run_id: str,
    mlflow_run_id: str,
    publication_id: Optional[str] = None,
    versions: Optional[Dict[str, str]] = None,
) -> str:
    """Génère la représentation textuelle normalisée de la chaîne de corrélation."""
    pub_part = publication_id or "non_publie"
    ver_part = str(versions) if versions else "{}"
    return (
        f"[CORRELATION] batch_id={batch_id} -> "
        f"flow_run_id={flow_run_id} -> "
        f"mlflow_run_id={mlflow_run_id} -> "
        f"publication_id={pub_part} -> "
        f"versions={ver_part}"
    )


def build_correlation_record(
    batch_id: str,
    flow_run_id: str,
    mlflow_run_id: str,
    publication_id: Optional[str] = None,
    versions: Optional[Dict[str, str]] = None,
    status: str = "COMPLETED",
) -> Dict[str, Any]:
    """Construit un dictionnaire structuré et sérialisable de la corrélation."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "batch_id": batch_id,
        "flow_run_id": flow_run_id,
        "mlflow_run_id": mlflow_run_id,
        "publication_id": publication_id,
        "model_versions": versions or {},
        "status": status,
        "formatted_chain": format_correlation_chain(
            batch_id=batch_id,
            flow_run_id=flow_run_id,
            mlflow_run_id=mlflow_run_id,
            publication_id=publication_id,
            versions=versions,
        ),
    }
