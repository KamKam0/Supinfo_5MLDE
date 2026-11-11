#!/usr/bin/env python3
"""scripts/select_models.py - Sélection, promotion vers champion et retour arrière.

Usage :
    uv run --locked python scripts/select_models.py --selection state/requested-selection.json
    uv run --locked python scripts/select_models.py --rollback 1 --reason "Incident de dérive sur v2"
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow
from mlflow.tracking import MlflowClient

from src.settings import settings
from src.tracking.publications import PublicationManager
from src.tracking.reconciliation import find_existing_model_version_for_run
from src.tracking.registry import (
    get_client,
    get_model_version_by_alias,
    promote_model_to_alias,
    rollback_model_alias,
    set_model_alias,
)

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def execute_promotion(
    selection_file: str | Path,
    target_alias: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    pub_manager: Optional[PublicationManager] = None,
) -> int:
    """Exécute la promotion formelle d'un couple de modèles selon l'arbitrage enregistré."""
    sel_path = Path(selection_file)
    if not sel_path.exists():
        logger.error(f"Fichier de sélection introuvable : {sel_path}")
        return 1

    with open(sel_path, "r", encoding="utf-8") as f:
        sel = json.load(f)

    run_id = sel.get("candidate_run_id")
    verdict = sel.get("verdict")
    requested_alias = (target_alias or sel.get("target_alias", "champion")).lstrip("@")

    print("=" * 80)
    print("      EXÉCUTION DE LA PROMOTION / SÉLECTION DANS LE REGISTRE MLFLOW")
    print("=" * 80)
    print(f"Run ID Candidat : {run_id}")
    print(f"Verdict         : {verdict}")
    print(f"Alias Cible     : @{requested_alias}")
    print("-" * 80)

    if verdict == "REJECTED":
        print(f"[REJET STRICT] Le run {run_id} a été rejeté par la politique de qualification.")
        print("Toute promotion vers l'alias champion est formellement bloquée.")
        return 1

    if verdict == "DEMO_ONLY" and requested_alias == "champion":
        print("[BLOCAGE PROVENANCE] Ce modèle est issu de flux synthétiques (DEMO_ONLY).")
        print("Promotion déclassée automatiquement vers l'alias '@candidate'.")
        requested_alias = "candidate"

    cli = get_client(tracking_uri)
    manager = pub_manager or PublicationManager()

    # Création du jalon de publication durable
    reg_name = sel.get("models", {}).get("regression", "steam-positive-ratio")
    cls_name = sel.get("models", {}).get("classification", "steam-owner-range")

    reg_ver = find_existing_model_version_for_run(cli, reg_name, run_id)
    cls_ver = find_existing_model_version_for_run(cli, cls_name, run_id)

    if not reg_ver or not cls_ver:
        logger.error(f"Impossible de localiser les versions enregistrées pour le run {run_id} dans MLflow.")
        return 1

    pub_id = manager.create_publication(
        run_id=run_id,
        verdict=verdict,
        target_alias=requested_alias,
        regression_model_name=reg_name,
        classification_model_name=cls_name,
        regression_model_uri=f"runs:/{run_id}/model_regression",
        classification_model_uri=f"runs:/{run_id}/model_classification",
    )
    manager.update_publication(
        pub_id,
        status="IN_PROGRESS",
        regression_version=reg_ver,
        classification_version=cls_ver,
    )

    # 1. Promotion Régression
    reg_promo = promote_model_to_alias(
        model_name=reg_name,
        version=reg_ver,
        alias=requested_alias,
        reason=f"Promotion formelle ({pub_id})",
        triggered_by="selection_script",
        qualification_status=verdict.lower(),
        pub_manager=manager,
        client=cli,
    )
    manager.update_publication(pub_id, regression_status="ALIASED")
    print(f"✅ Modèle Régression '{reg_name}' : v{reg_ver} -> @{requested_alias} (Evt: {reg_promo['event_id']})")

    # 2. Promotion Classification
    cls_promo = promote_model_to_alias(
        model_name=cls_name,
        version=cls_ver,
        alias=requested_alias,
        reason=f"Promotion formelle ({pub_id})",
        triggered_by="selection_script",
        qualification_status=verdict.lower(),
        pub_manager=manager,
        client=cli,
    )
    manager.update_publication(pub_id, classification_status="ALIASED", status="COMPLETED")
    print(f"✅ Modèle Classification '{cls_name}' : v{cls_ver} -> @{requested_alias} (Evt: {cls_promo['event_id']})")

    # 3. Mise à jour du manifeste local (state/local-models.json)
    state_manifest = Path("state") / "local-models.json"
    if state_manifest.exists():
        with open(state_manifest, "r", encoding="utf-8") as f:
            local_meta = json.load(f)
        local_meta["champion_alias"] = requested_alias
        local_meta["regression_champion_version"] = reg_ver
        local_meta["classification_champion_version"] = cls_ver
        local_meta["last_promotion_at"] = datetime.now(timezone.utc).isoformat()
        with open(state_manifest, "w", encoding="utf-8") as f:
            json.dump(local_meta, f, indent=2)

    print("-" * 80)
    print(f"PROMOTION RÉUSSIE : Publication {pub_id} finalisée avec succès.")
    print("=" * 80)
    return 0


def execute_rollback(
    target_version: str,
    reason: str,
    alias: str = "champion",
    tracking_uri: Optional[str] = None,
    pub_manager: Optional[PublicationManager] = None,
) -> int:
    """Exécute la procédure de retour arrière (rollback) vers une version antérieure."""
    clean_alias = alias.lstrip("@")
    cli = get_client(tracking_uri)
    manager = pub_manager or PublicationManager()

    print("=" * 80)
    print("      PROCÉDURE D'URGENCE DE RETOUR ARRIÈRE (ROLLBACK)")
    print("=" * 80)
    print(f"Version cible à restaurer : {target_version}")
    print(f"Alias affecté             : @{clean_alias}")
    print(f"Motif du retour arrière   : {reason}")
    print("-" * 80)

    reg_name = "steam-positive-ratio"
    cls_name = "steam-owner-range"

    # Rollback Régression
    reg_res = rollback_model_alias(
        model_name=reg_name,
        target_version=target_version,
        alias=clean_alias,
        reason=reason,
        triggered_by="rollback_cli",
        pub_manager=manager,
        client=cli,
    )
    print(f"🔄 Régression '{reg_name}' : @{clean_alias} réassigné à v{target_version} (précédent: v{reg_res['replaced_version']})")

    # Rollback Classification
    cls_res = rollback_model_alias(
        model_name=cls_name,
        target_version=target_version,
        alias=clean_alias,
        reason=reason,
        triggered_by="rollback_cli",
        pub_manager=manager,
        client=cli,
    )
    print(f"🔄 Classification '{cls_name}' : @{clean_alias} réassigné à v{target_version} (précédent: v{cls_res['replaced_version']})")

    # Mise à jour du manifeste local
    state_manifest = Path("state") / "local-models.json"
    if state_manifest.exists():
        with open(state_manifest, "r", encoding="utf-8") as f:
            local_meta = json.load(f)
        local_meta["champion_alias"] = clean_alias
        local_meta["regression_champion_version"] = target_version
        local_meta["classification_champion_version"] = target_version
        local_meta["last_rollback_at"] = datetime.now(timezone.utc).isoformat()
        local_meta["last_rollback_reason"] = reason
        with open(state_manifest, "w", encoding="utf-8") as f:
            json.dump(local_meta, f, indent=2)

    print("-" * 80)
    print(f"RETOUR ARRIÈRE ACHEVÉ : Version {target_version} active en production.")
    print("=" * 80)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Promotion ou Rollback des modèles dans le Model Registry MLflow.")
    parser.add_argument("--selection", type=str, default="state/requested-selection.json", help="Chemin du fichier de sélection")
    parser.add_argument("--alias", type=str, default="champion", help="Alias à attribuer (défaut: champion)")
    parser.add_argument("--rollback", type=str, default=None, help="Version cible vers laquelle effectuer un rollback")
    parser.add_argument("--reason", type=str, default="Intervention manuelle de l'opérateur", help="Motif du rollback")
    parser.add_argument("--url", type=str, default=None, help="URI de tracking MLflow")
    args = parser.parse_args()

    uri = args.url or os.environ.get("MLFLOW_TRACKING_URI", settings.MLFLOW_TRACKING_URI)

    if args.rollback is not None:
        return execute_rollback(
            target_version=args.rollback,
            reason=args.reason,
            alias=args.alias,
            tracking_uri=uri,
        )
    else:
        return execute_promotion(
            selection_file=args.selection,
            target_alias=args.alias,
            tracking_uri=uri,
        )


if __name__ == "__main__":
    sys.exit(main())
