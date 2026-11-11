#!/usr/bin/env python3
"""scripts/compare_runs.py - Comparaison de runs MLflow et application de la politique de qualification.

Usage :
    uv run --locked python scripts/compare_runs.py --config config/qualification.json
    uv run --locked python scripts/compare_runs.py --run-id <run_id>
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow
from mlflow.tracking import MlflowClient

from src.settings import settings
from src.tracking.qualification import evaluate_run_qualification, load_qualification_policy

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"


def get_run_data(client: MlflowClient, run_id: str) -> Dict[str, Any]:
    """Extrait les métriques et tags d'un run MLflow."""
    run = client.get_run(run_id)
    return {
        "run_id": run_id,
        "experiment_id": run.info.experiment_id,
        "status": run.info.status,
        "metrics": run.data.metrics,
        "params": run.data.params,
        "tags": run.data.tags,
    }


def find_latest_active_run(client: MlflowClient, experiment_name: str = "steam-mlops-experiments") -> str:
    """Trouve l'ID du run le plus récent dans l'expérience MLflow."""
    exp = client.get_experiment_by_name(experiment_name)
    if not exp:
        raise ValueError(f"Expérience '{experiment_name}' introuvable.")
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
        max_results=5,
    )
    if not runs:
        raise ValueError(f"Aucun run trouvé dans l'expérience '{experiment_name}'.")
    return runs[0].info.run_id


def compare_and_qualify(
    candidate_run_id: str,
    baseline_run_id: Optional[str] = None,
    config_path: Optional[str | Path] = None,
    output_path: Optional[str | Path] = "state/requested-selection.json",
    tracking_uri: Optional[str] = None,
) -> int:
    """Compare un run candidat à sa baseline et applique la qualification déterministe."""
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", settings.MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=uri)

    print("=" * 80)
    print("      ARBITRAGE FORMEL DE QUALIFICATION DES MODÈLES STEAM")
    print("=" * 80)

    candidate = get_run_data(client, candidate_run_id)
    cand_metrics = candidate["metrics"]
    cand_tags = candidate["tags"]

    print(f"[MLFLOW] Analyse du Run Candidat : {candidate_run_id}")
    print(f"  Mode           : {cand_tags.get('pipeline_mode', 'UNKNOWN')}")
    print(f"  Snapshot SHA   : {cand_tags.get('snapshot_sha256', 'N/A')}")
    print(f"  Git Commit     : {cand_tags.get('git_commit', 'N/A')}")
    print("-" * 80)

    # Évaluation de la qualification
    report = evaluate_run_qualification(
        run_id=candidate_run_id,
        metrics=cand_metrics,
        tags=cand_tags,
        reload_test_passed=True,
        config_path=config_path,
    )

    # Affichage du tableau de contrôle
    print(f"{'RÈGLE DE QUALIFICATION':<35} | {'ATTENDU':<18} | {'OBSERVÉ':<14} | STATUT")
    print("-" * 80)
    for check in report.checks:
        status_label = "✅ PASS" if check.passed else "❌ FAIL"
        obs_display = str(check.observed)[:14]
        exp_display = str(check.expected)[:18]
        print(f"{check.name:<35} | {exp_display:<18} | {obs_display:<14} | {status_label}")
    print("-" * 80)

    # Métriques clés
    reg_cand_mae = cand_metrics.get("regression.val.mae", 0.0)
    reg_base_mae = cand_metrics.get("regression.val.baseline_mae", 0.0)
    cls_cand_acc = cand_metrics.get("classification.val.accuracy", 0.0)
    cls_base_acc = cand_metrics.get("classification.val.baseline_accuracy", 0.0)
    cls_cand_f1 = cand_metrics.get("classification.val.f1_macro", 0.0)
    cls_base_f1 = cand_metrics.get("classification.val.baseline_f1_macro", 0.0)

    print("SYNTHÈSE COMPARATIVE CANDIDAT VS BASELINE (VALIDATION) :")
    print(f"  Régression MAE       : {reg_cand_mae:.2f}%  (Baseline: {reg_base_mae:.2f}%) -> Diff: {reg_cand_mae - reg_base_mae:+.2f}%")
    print(f"  Classification Acc   : {cls_cand_acc:.2%} (Baseline: {cls_base_acc:.2%}) -> Diff: {cls_cand_acc - cls_base_acc:+.2%}")
    print(f"  Classification F1    : {cls_cand_f1:.4f}  (Baseline: {cls_base_f1:.4f}) -> Diff: {cls_cand_f1 - cls_base_f1:+.4f}")
    print("-" * 80)

    # Verdict
    print(f"VERDICT OFFICIEL : [{report.verdict}]")
    print(f"Résumé           : {report.summary}")
    if report.rejection_reasons:
        print("Motifs de rejet  :")
        for r in report.rejection_reasons:
            print(f"  - {r}")
    print("=" * 80)

    # Sauvegarde de la demande de sélection
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        selection_data = {
            "candidate_run_id": candidate_run_id,
            "verdict": report.verdict,
            "passed": report.passed,
            "target_alias": "champion" if report.verdict == "PRODUCTION_READY" else "candidate",
            "models": {
                "regression": "steam-positive-ratio",
                "classification": "steam-owner-range",
            },
            "metrics": {
                "regression_val_mae": reg_cand_mae,
                "regression_val_baseline_mae": reg_base_mae,
                "classification_val_acc": cls_cand_acc,
                "classification_val_baseline_acc": cls_base_acc,
                "classification_val_f1": cls_cand_f1,
                "classification_val_baseline_f1": cls_base_f1,
            },
            "tags": cand_tags,
            "timestamp": report.timestamp,
        }
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(selection_data, f, indent=2)
        print(f"[ARBITRAGE] Demande de sélection enregistrée dans : {out_p}")

    return 0 if report.passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Arbitrage formel de qualification d'un modèle.")
    parser.add_argument("--config", type=str, default="config/qualification.json", help="Chemin vers qualification.json")
    parser.add_argument("--run-id", type=str, default=None, help="Run ID MLflow à évaluer (None = dernier run)")
    parser.add_argument("--output", type=str, default="state/requested-selection.json", help="Fichier de sélection généré")
    parser.add_argument("--url", type=str, default=None, help="URI MLflow")
    args = parser.parse_args()

    uri = args.url or os.environ.get("MLFLOW_TRACKING_URI", settings.MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=uri)

    run_id = args.run_id or find_latest_active_run(client)
    return compare_and_qualify(
        candidate_run_id=run_id,
        config_path=args.config,
        output_path=args.output,
        tracking_uri=uri,
    )


if __name__ == "__main__":
    sys.exit(main())
