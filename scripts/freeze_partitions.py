#!/usr/bin/env python3
"""scripts/freeze_partitions.py - Script de figeage déterministe des partitions sans fuite.

Lit le snapshot projeté (data/snapshots/historical_projection.csv),
applique le partitionnement stratifié (70% Train, 15% Val, 15% Test),
contrôle l'étanchéité mathématique et persiste les identifiants dans data/partitions/*.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import pandas as pd

# Ajout de src au path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.partitions import build_stratified_partitions, save_partitions_to_json, verify_partitions_integrity
from src.settings import settings


def freeze_partitions(config_path: str | Path) -> dict:
    conf_file = Path(config_path).resolve()
    if not conf_file.exists():
        raise FileNotFoundError(f"Configuration introuvable : {conf_file}")

    with open(conf_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    snapshot_path_raw = config.get("snapshot_path", "data/snapshots/historical_projection.csv")
    snapshot_path = project_root / snapshot_path_raw if not Path(snapshot_path_raw).is_absolute() else Path(snapshot_path_raw)

    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot historique introuvable à : {snapshot_path}")

    print(f"Chargement du snapshot historique : {snapshot_path}")
    df = pd.read_csv(snapshot_path)
    total_rows = len(df)
    print(f"Total jeux à partitionner : {total_rows:,}")

    seed = config.get("random_seed", settings.RANDOM_SEED)
    print(f"Exécution du partitionnement déterministe (graine = {seed})...")

    partitions = build_stratified_partitions(df, seed=seed)

    train_count = len(partitions["train_appids"])
    val_count = len(partitions["val_appids"])
    test_count = len(partitions["test_appids"])
    reserved_count = len(partitions["reserved_appids"])

    print("-" * 60)
    print(f"  Train (70%)         : {train_count:,} appids ({train_count / total_rows:.2%})")
    print(f"  Validation (15%)    : {val_count:,} appids ({val_count / total_rows:.2%})")
    print(f"  Test Final (15%)    : {test_count:,} appids ({test_count / total_rows:.2%})")
    print(f"  Réservés (Val+Test) : {reserved_count:,} appids ({reserved_count / total_rows:.2%})")
    print("-" * 60)

    # Sauvegarde des fichiers JSON
    output_dir = project_root / "data" / "partitions"
    save_partitions_to_json(partitions, output_dir)
    print(f"[SUCCÈS] Partitions enregistrées dans : {output_dir}")

    # Contrôle d'intégrité
    verify_partitions_integrity(
        partitions["train_appids"],
        partitions["val_appids"],
        partitions["test_appids"],
        total_expected=total_rows,
    )
    print("[INTÉGRITÉ VALIDÉE] Aucune intersection détectée entre Train, Validation et Test.")

    return {
        "train_count": train_count,
        "val_count": val_count,
        "test_count": test_count,
        "reserved_count": reserved_count,
        "total_rows": total_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Figeage des partitions déterministes Steam.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/training.historical.json",
        help="Chemin du fichier de configuration (défaut: config/training.historical.json)",
    )
    args = parser.parse_args()

    try:
        freeze_partitions(args.config)
        return 0
    except Exception as exc:
        print(f"Erreur lors du partitionnement : {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
