#!/usr/bin/env python3
"""main.py - Point d'entrée CLI pour le pipeline d'entraînement MLOps Steam.

Enchaîne séquentiellement les 4 fonctions mères sans aucune dépendance envers Jupyter :
1. load_data(config_path)
2. preprocess_data(ingestion_result)
3. train_models(prepared_data)
4. evaluate_models(models_bundle, prepared_data)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import pandas as pd

from src.evaluation import evaluate_models
from src.ingestion import load_data
from src.preprocessing import preprocess_data
from src.training import train_models


def run_pipeline(config_path: str | Path) -> int:
    """Exécute de bout en bout la chaîne d'entraînement modulaire."""
    t_start = time.time()
    print("=" * 70)
    print("    PIPELINE MODULAIRE MLOPS STEAM - ENTRAÎNEMENT & QUALIFICATION")
    print("=" * 70)

    # 1. Ingestion
    ingestion_result = load_data(config_path)

    # 2. Prétraitement & Éligibilité
    prepared_data = preprocess_data(ingestion_result)

    # Lecture de la source pour les étapes d'entraînement et d'évaluation
    full_df = pd.read_csv(ingestion_result.snapshot.path)

    # 3. Entraînement
    models_bundle = train_models(prepared_data, full_df)

    # 4. Évaluation & Sérialisation
    evaluate_models(models_bundle, prepared_data, full_df, artifacts_dir="artifacts")

    total_time = time.time() - t_start
    print(f"[PIPELINE ACHEVÉ] Succès complet en {total_time:.2f}s.")
    print("=" * 70)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Pipeline MLOps Steam d'entraînement.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/training.historical.json",
        help="Chemin du fichier de configuration (défaut: config/training.historical.json)",
    )
    args = parser.parse_args()

    try:
        return run_pipeline(args.config)
    except Exception as exc:
        print(f"[ERREUR FATALE] Échec du pipeline : {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
