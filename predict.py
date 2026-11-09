#!/usr/bin/env python3
"""predict.py - CLI d'inférence locale autonome sur modèles sérialisés.

Charge les pipelines sérialisés via le manifeste local,
valide les entrées par le contrat strict InferenceInputGame (anti-fuite),
exécute les prédictions des deux modèles et exporte les résultats en CSV.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import joblib
import pandas as pd

from src.contracts import InferenceInputGame
from src.settings import settings


def run_prediction(manifest_path: str | Path, input_csv: str | Path, output_csv: str | Path) -> int:
    m_path = Path(manifest_path).resolve()
    in_path = Path(input_csv).resolve()
    out_path = Path(output_csv).resolve()

    if not m_path.exists():
        raise FileNotFoundError(f"Manifeste de modèles introuvable : {m_path}")
    if not in_path.exists():
        raise FileNotFoundError(f"Fichier d'entrée introuvable : {in_path}")

    with open(m_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    reg_path = Path(manifest["regression_model_path"])
    cls_path = Path(manifest["classification_model_path"])

    if not reg_path.exists():
        raise FileNotFoundError(f"Pipeline régression introuvable : {reg_path}")
    if not cls_path.exists():
        raise FileNotFoundError(f"Pipeline classification introuvable : {cls_path}")

    print(f"[PREDICT] Chargement modèle régression    : {reg_path.name}")
    reg_pipe = joblib.load(reg_path)
    print(f"[PREDICT] Chargement modèle classification: {cls_path.name}")
    cls_pipe = joblib.load(cls_path)

    # 1. Lecture et validation stricte des entrées
    print(f"[PREDICT] Lecture des entrées depuis : {in_path}")
    raw_df = pd.read_csv(in_path)

    # Validation Pydantic ligne par ligne pour interdire formellement toute fuite
    validated_rows = []
    for idx, row_dict in enumerate(raw_df.to_dict(orient="records")):
        # Nettoyage des NaN pour compatibilité Pydantic
        clean_dict = {k: v for k, v in row_dict.items() if pd.notna(v)}
        try:
            validated_item = InferenceInputGame(**clean_dict)
            validated_rows.append(validated_item.model_dump())
        except Exception as exc:
            raise ValueError(f"Ligne {idx} non conforme au contrat d'inférence : {exc}")

    eval_df = pd.DataFrame(validated_rows)
    feature_cols = list(settings.ALLOWED_INFERENCE_FEATURES)
    X = eval_df[feature_cols]

    # 2. Exécution des prédictions
    pct_preds = reg_pipe.predict(X)
    owners_preds = cls_pipe.predict(X)

    results_df = pd.DataFrame({
        "appid": eval_df["appid"],
        "predicted_pct_pos_total": [round(float(p), 2) for p in pct_preds],
        "predicted_estimated_owners": owners_preds,
    })

    # 3. Export
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)
    print(f"[PREDICT] {len(results_df)} prédictions écrites dans : {out_path}")
    print("-" * 60)
    print(results_df.to_string(index=False))
    print("-" * 60)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Inférence locale CLI Steam MLOps.")
    parser.add_argument(
        "--manifest",
        type=str,
        default="state/local-models.json",
        help="Chemin du manifeste de modèles (défaut: state/local-models.json)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/examples/inference.csv",
        help="Chemin du fichier CSV d'entrée sans cibles",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/inference-results.csv",
        help="Chemin du fichier CSV de sortie pour les prédictions",
    )
    args = parser.parse_args()

    try:
        return run_prediction(args.manifest, args.input, args.output)
    except Exception as exc:
        print(f"[ERREUR] Échec de l'inférence : {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
