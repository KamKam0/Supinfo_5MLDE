"""api_trigger_passif/worker.py - Worker d'arrière-plan traitant les lots durables.

Scrute les lots à l'état PENDING dans SQLite, déclenche la validation GX,
l'upsert et l'entraînement, et met à jour leur statut (SUCCEEDED / FAILED).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import time
from typing import Optional

from api_trigger_passif.storage import PushStorage
from api_trigger_passif.training_trigger import process_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def process_pending_batches(
    storage: PushStorage,
    run_once: bool = False,
    poll_interval: float = 2.0,
    trigger_training: bool = False,
) -> int:
    """Traite les lots en attente dans la base SQLite."""
    processed_count = 0
    while True:
        status_info = storage.get_status()
        pending_batches = [
            b for b in status_info.get("recent_batches", []) if b["status"] == "PENDING"
        ]

        for b in pending_batches:
            batch_id = b["batch_id"]
            logger.info(f"[WORKER] Traitement du lot en attente : {batch_id}")
            result = process_batch(
                batch_id=batch_id,
                storage=storage,
                trigger_training=trigger_training,
            )
            logger.info(f"[WORKER] Lot {batch_id} terminé avec statut : {result.get('status')}")
            processed_count += 1

        if run_once or not pending_batches:
            if run_once:
                break
            time.sleep(poll_interval)

    return processed_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Worker d'arrière-plan de traitement des lots PUSH")
    parser.add_argument("--once", action="store_true", help="Traiter les lots existants une seule fois et quitter")
    parser.add_argument("--db-path", type=str, default=None, help="Chemin vers la base de données SQLite")
    parser.add_argument("--train", action="store_true", help="Déclencher l'entraînement des modèles")
    parser.add_argument("--interval", type=float, default=2.0, help="Intervalle de scrutation en secondes")

    args = parser.parse_args()
    storage = PushStorage(db_path=args.db_path)

    logger.info("Démarrage du worker PUSH...")
    process_pending_batches(
        storage=storage,
        run_once=args.once,
        poll_interval=args.interval,
        trigger_training=args.train,
    )


if __name__ == "__main__":
    main()
