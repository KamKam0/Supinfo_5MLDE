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

    # 4. Évaluation & Sérialisation locale
    eval_metrics = evaluate_models(models_bundle, prepared_data, full_df, artifacts_dir="artifacts")

    # 5. Suivi & Enregistrement MLflow
    try:
        from src.tracking.mlflow_tracking import log_training_run
        mlflow_info = log_training_run(
            models_bundle=models_bundle,
            prepared_data=prepared_data,
            eval_metrics=eval_metrics,
            sample_df=full_df,
            snapshot=ingestion_result.snapshot,
            register_models=True,
        )
        print(f"[MLFLOW] Run enregistré avec succès : ID={mlflow_info['mlflow_run_id']}")
        print(f"  Modèle Régression     : {mlflow_info['regression_model_uri']}")
        print(f"  Modèle Classification : {mlflow_info['classification_model_uri']}")
        
        # Mise à jour du manifeste local avec les références MLflow
        state_manifest = Path("state") / "local-models.json"
        if state_manifest.exists():
            import json
            with open(state_manifest, "r", encoding="utf-8") as f:
                s_data = json.load(f)
            s_data["mlflow_run_id"] = mlflow_info["mlflow_run_id"]
            s_data["mlflow_experiment_id"] = mlflow_info["experiment_id"]
            s_data["mlflow_regression_uri"] = mlflow_info["regression_model_uri"]
            s_data["mlflow_classification_uri"] = mlflow_info["classification_model_uri"]
            with open(state_manifest, "w", encoding="utf-8") as f:
                json.dump(s_data, f, indent=2)
    except Exception as exc:
        print(f"[MLFLOW AVERTISSEMENT] Suivi MLflow ignoré ou indisponible : {exc}")

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
