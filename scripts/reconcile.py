#!/usr/bin/env python3
"""scripts/reconcile.py - Réconciliation et reprise après panne des publications.

Usage :
    uv run --locked python scripts/reconcile.py --publication-id test-pub-01
    uv run --locked python scripts/reconcile.py --scan-pending
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.settings import settings
from src.tracking.publications import PublicationManager
from src.tracking.reconciliation import reconcile_publication

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def reconcile_single_publication(pub_id: str, tracking_uri: Optional[str] = None) -> int:
    """Réconcilie une publication spécifique."""
    print("=" * 80)
    print(f"      RÉCONCILIATION DE PUBLICATION INTERROMPUE : {pub_id}")
    print("=" * 80)
    try:
        res = reconcile_publication(pub_id, tracking_uri=tracking_uri)
        print(f"Statut final              : {res['status']}")
        print(f"Version Régression        : v{res['regression_version']}")
        print(f"Version Classification    : v{res['classification_version']}")
        print(f"Jalons repris / exécutés  : {res['reconciled_steps']}")
        print("=" * 80)
        return 0
    except Exception as exc:
        print(f"[ERREUR RÉCONCILIATION] Échec sur {pub_id} : {exc}", file=sys.stderr)
        print("=" * 80)
        return 1


def reconcile_all_pending(tracking_uri: Optional[str] = None) -> int:
    """Scanne et réconcilie toutes les publications non complétées."""
    manager = PublicationManager()
    pubs = manager.list_publications()
    non_completed = [p for p in pubs if p["status"] != "COMPLETED"]

    if not non_completed:
        print("[RÉCONCILIATION] Aucune publication interrompue ou en attente détectée.")
        return 0

    print(f"[RÉCONCILIATION] {len(non_completed)} publication(s) non complétée(s) trouvée(s).")
    failures = 0
    for p in non_completed:
        pub_id = p["publication_id"]
        code = reconcile_single_publication(pub_id, tracking_uri=tracking_uri)
        if code != 0:
            failures += 1

    return 0 if failures == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Outil de réconciliation des publications interrompues.")
    parser.add_argument("--publication-id", type=str, default=None, help="Identifiant de la publication à réconcilier")
    parser.add_argument("--scan-pending", action="store_true", help="Scanner et réconcilier tous les jalons en attente")
    parser.add_argument("--url", type=str, default=None, help="URI de tracking MLflow")
    args = parser.parse_args()

    uri = args.url or os.environ.get("MLFLOW_TRACKING_URI", settings.MLFLOW_TRACKING_URI)

    if args.publication_id:
        return reconcile_single_publication(args.publication_id, tracking_uri=uri)
    else:
        return reconcile_all_pending(tracking_uri=uri)


if __name__ == "__main__":
    sys.exit(main())
