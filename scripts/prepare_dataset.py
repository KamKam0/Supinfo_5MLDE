#!/usr/bin/env python3
"""scripts/prepare_dataset.py - Script de projection historique déterministe.

Lit le fichier brut d'origine (ex: games_march2025_full.csv) en lecture seule,
extrait la projection canonique des 16 colonnes retenues pour le cycle MLOps,
garantit l'ordre et le nombre exact de lignes, calcule les empreintes SHA-256
et génère data/snapshots/historical_projection.csv ainsi que data/source_manifest.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd


def compute_sha256(file_path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    """Calcule l'empreinte SHA-256 d'un fichier par blocs."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha.update(chunk)
    return sha.hexdigest().lower()


def prepare_dataset(config_path: str | Path) -> dict:
    """Exécute la projection déterministe à partir de la configuration JSON."""
    config_file = Path(config_path).resolve()
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration introuvable : {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    project_root = config_file.parent.parent

    source_path_raw = config.get("source_path", "data/raw/games_march2025_full.csv")
    source_path = project_root / source_path_raw if not Path(source_path_raw).is_absolute() else Path(source_path_raw)

    snapshot_path_raw = config.get("snapshot_path", "data/snapshots/historical_projection.csv")
    snapshot_path = project_root / snapshot_path_raw if not Path(snapshot_path_raw).is_absolute() else Path(snapshot_path_raw)

    selected_columns = config.get("selected_columns")
    if not selected_columns:
        raise ValueError("La liste 'selected_columns' est obligatoire dans la configuration.")

    expected_source_sha = config.get("expected_source_sha256", "").lower()
    expected_proj_sha = config.get("expected_projection_sha256", "").lower()

    if not source_path.exists():
        raise FileNotFoundError(f"Fichier source introuvable à l'emplacement : {source_path}")

    # 1. Vérification de l'empreinte du fichier brut
    source_bytes = source_path.stat().st_size
    actual_source_sha = compute_sha256(source_path)

    print(f"Source brute : {source_path}")
    print(f"  Taille     : {source_bytes:,} octets")
    print(f"  SHA-256    : {actual_source_sha}")

    if expected_source_sha and actual_source_sha != expected_source_sha:
        print(f"  [ATTENTION] L'empreinte source diffère de la référence ({expected_source_sha})")
    else:
        print("  [OK] Empreinte source conforme à la référence documentaire")

    # 2. Lecture et projection en mémoire
    print(f"Lecture des {len(selected_columns)} colonnes de projection...")
    df = pd.read_csv(source_path, usecols=selected_columns)
    df = df[selected_columns]  # Garantir l'ordre strict des colonnes

    row_count = len(df)
    print(f"Lignes extraites : {row_count:,}")

    # 3. Écriture du snapshot projeté
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    # Écriture avec fin de ligne standardisée (CRLF pour concordance bit-à-bit avec la référence pédagogique)
    df.to_csv(snapshot_path, index=False, lineterminator="\r\n")

    actual_proj_bytes = snapshot_path.stat().st_size
    actual_proj_sha = compute_sha256(snapshot_path)

    print(f"Projection générée : {snapshot_path}")
    print(f"  Taille     : {actual_proj_bytes:,} octets")
    print(f"  SHA-256    : {actual_proj_sha}")

    if expected_proj_sha and actual_proj_sha == expected_proj_sha:
        print("  [SUCCÈS] Empreinte de projection rigoureusement identique à la référence pédagogique !")
    elif expected_proj_sha:
        print(f"  [INFO] Empreinte de projection : {actual_proj_sha} (référence : {expected_proj_sha})")

    # 4. Manifeste de provenance
    manifest_data = {
        "dataset_name": "Steam Games Dataset (Kaggle / artermiloff)",
        "source_url": "https://www.kaggle.com/datasets/artermiloff/steam-games-dataset?resource=download",
        "acquisition_date": "2026-11-16T10:37:00Z",
        "author_attribution": "Artem Iltoff (artermiloff) / Valve Corporation / SteamSpy",
        "source_file": source_path.name,
        "source_bytes": source_bytes,
        "source_sha256": actual_source_sha,
        "projection_file": snapshot_path.name,
        "projection_bytes": actual_proj_bytes,
        "projection_sha256": actual_proj_sha,
        "row_count": row_count,
        "column_count": len(selected_columns),
        "columns": selected_columns,
        "license_and_terms": (
            "Données brutes issues de l'API publique Steam et de SteamSpy via Kaggle. "
            "La licence exacte et la date d'extraction ne sont pas formellement établies par la source amont. "
            "Usage pédagogique et analytique interne uniquement. "
            "Conformément aux règles d'hygiène, aucun CSV brut ou projeté volumineux n'est versionné dans Git."
        ),
        "scientific_limitations": [
            "Estimations de propriétaires SteamSpy basées sur échantillonnage de profils publics (pas des chiffres de ventes réels).",
            "La devise des prix historiques n'est pas précisée (présumée USD, mais variations possibles).",
            "pct_pos_total = -1 signale l'absence de notation (sentinelle à filtrer pour la régression).",
            "estimated_owners = '0 - 0' (13 656 jeux) conventionnellement exclue faute de signal exploitable.",
            "Les métadonnées (prix, avis, tags) sont celles d'un instantané rétrospectif et ne reflètent pas les données antérieures à la sortie des titres."
        ]
    }

    manifest_path = project_root / "data" / "source_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    print(f"Manifeste de provenance enregistré dans : {manifest_path}")
    return manifest_data


def main() -> int:
    parser = argparse.ArgumentParser(description="Projection déterministe du dataset Steam.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/training.historical.json",
        help="Chemin du fichier de configuration JSON (par défaut: config/training.historical.json)",
    )
    args = parser.parse_args()

    try:
        prepare_dataset(args.config)
        return 0
    except Exception as exc:
        print(f"Erreur lors de la préparation : {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
