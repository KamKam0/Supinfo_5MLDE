"""api_trigger_passif/training_trigger.py - Déclenchement d'entraînement et validation de lot.

Gère :
- L'extraction du snapshot figé de 500 jeux depuis SQLite.
- La validation stricte du lot via la suite Great Expectations push_batch_v1.
- La fusion par upsert avec le snapshot de base.
- L'entraînement et la mise à jour durable du statut du lot (SUCCEEDED ou FAILED).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd

from api_trigger_passif.storage import PushStorage
from src.data_sources.snapshots import upsert_and_publish_snapshot
from src.data_validation import validate_dataframe
from src.exceptions import DataValidationError
from src.training import train_models

logger = logging.getLogger(__name__)


def process_batch(
    batch_id: str,
    storage: PushStorage,
    base_snapshot_path: Optional[Path | str] = None,
    snapshots_dir: Optional[Path | str] = None,
    trigger_training: bool = False,
) -> Dict[str, Any]:
    """Exécute la chaîne de traitement d'un lot : validation GX, publication de snapshot et entraînement.

    Args:
        batch_id: Identifiant du lot à traiter.
        storage: Gestionnaire de persistance SQLite.
        base_snapshot_path: Chemin du snapshot historique source.
        snapshots_dir: Répertoire de publication du nouveau snapshot.
        trigger_training: Si True, lance l'entraînement complet des modèles.

    Returns:
        Dictionnaire récapitulatif du traitement du lot.
    """
    project_root = Path(__file__).parent.parent
    base_path = Path(base_snapshot_path or (project_root / "data" / "snapshots" / "historical_projection.csv")).resolve()
    snap_dir = Path(snapshots_dir or (project_root / "data" / "snapshots")).resolve()
    snap_dir.mkdir(parents=True, exist_ok=True)

    storage.update_batch_status(batch_id, "RUNNING")

    # 1. Extraction des 500 jeux composant le lot
    batch_df = storage.get_batch_games_df(batch_id)
    if batch_df is None or len(batch_df) == 0:
        error_msg = f"Aucun jeu trouvé pour le lot '{batch_id}'."
        storage.update_batch_status(batch_id, "FAILED", failure_reason=error_msg)
        return {"batch_id": batch_id, "status": "FAILED", "error": error_msg}

    # 2. Barrière bloquante Great Expectations : suite push_batch_v1
    try:
        report = validate_dataframe(
            batch_df,
            suite_name="push_batch_v1",
            dataset_name=batch_id,
            raise_on_error=True,
        )
    except DataValidationError as exc:
        error_msg = f"Rejet par Great Expectations (push_batch_v1) : {exc}"
        logger.error(f"[BATCH] {error_msg}")
        storage.update_batch_status(batch_id, "FAILED", failure_reason=str(exc))
        return {"batch_id": batch_id, "status": "FAILED", "error": str(exc)}

    # 3. Upsert et publication du nouveau snapshot immuable
    output_snapshot = snap_dir / f"historical_with_push_{batch_id}.csv"
    try:
        summary = upsert_and_publish_snapshot(
            base_snapshot_path=base_path,
            new_observations=batch_df,
            output_snapshot_path=output_snapshot,
            reserved_appids_path=project_root / "data" / "partitions" / "reserved_appids.json",
            validate_before_merge=False,  # Déjà validé par push_batch_v1 ci-dessus
            validate_after_merge=True,
        )
    except Exception as exc:
        error_msg = f"Échec d'upsert et de validation de snapshot : {exc}"
        logger.error(f"[BATCH] {error_msg}")
        storage.update_batch_status(batch_id, "FAILED", failure_reason=error_msg)
        return {"batch_id": batch_id, "status": "FAILED", "error": error_msg}

    # 4. Déclenchement de l'entraînement si demandé
    if trigger_training:
        try:
            from src.ingestion import load_data
            from src.preprocessing import preprocess_data
            from src.evaluation import evaluate_models

            # Préparation de configuration d'entraînement
            ing_res = load_data(
                {
                    "snapshot_path": str(output_snapshot),
                    "validation_suite": "snapshot_integrity_v1",
                }
            )
            prep_data = preprocess_data(ing_res)
            # Entraînement
            train_models(prep_data)
        except Exception as exc:
            error_msg = f"Échec lors de l'entraînement du lot : {exc}"
            logger.error(f"[BATCH] {error_msg}")
            storage.update_batch_status(
                batch_id,
                "FAILED",
                snapshot_path=str(output_snapshot),
                snapshot_sha256=summary["new_snapshot_sha256"],
                failure_reason=error_msg,
            )
            return {"batch_id": batch_id, "status": "FAILED", "error": error_msg}

    # 5. Succès durable
    storage.update_batch_status(
        batch_id,
        "SUCCEEDED",
        snapshot_path=str(output_snapshot),
        snapshot_sha256=summary["new_snapshot_sha256"],
    )

    print(
        f"[BATCH] Lot {batch_id} traité avec succès : {summary['final_row_count']:,} jeux consolidés "
        f"(Snapshot: {output_snapshot.name}, SHA256: {summary['new_snapshot_sha256'][:12]}...)"
    )

    return {
        "batch_id": batch_id,
        "status": "SUCCEEDED",
        "snapshot_path": str(output_snapshot),
        "snapshot_sha256": summary["new_snapshot_sha256"],
        "row_count": summary["final_row_count"],
    }
