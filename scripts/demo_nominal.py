#!/usr/bin/env python3
"""scripts/demo_nominal.py - Démonstrateur nominal de bout en bout et soutenance MLOps.

Déroule chronologiquement le scénario nominal de référence en 7 étapes séquentielles :
1. Diagnostic de santé environnementale et services Docker (PostgreSQL, MLflow, Prefect, Grafana).
2. Validation des données brutes par la barrière Great Expectations Core 1.x (zéro anomalie).
3. Entraînement orchestré sous Prefect 3 avec tracking MLflow (métriques, artefacts, versions).
4. Qualification formelle des modèles et intégrité cryptographique du manifeste de service SHA-256.
5. Inférences unitaire et par lot (batch vectorisé) avec contrôle anti-fuite de cible (HTTP 422).
6. Soumission de retour utilisateur (/feedback) avec preuve formelle d'idempotence (201 -> 200).
7. Réconciliation comptable SQL à l'unité près (11 indicateurs alignés à 100%).

Option --with-incidents / --all :
Enchaîne avec le banc d'injection d'incidents (demo_incident.py) pour une démonstration intégrée totale.

Usage :
    uv run --locked python scripts/demo_nominal.py --campaign demo-nominal
    uv run --locked python scripts/demo_nominal.py --campaign demo-nominal --fast
    uv run --locked python scripts/demo_nominal.py --campaign demo-nominal --with-incidents
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("demo_nominal")

import pandas as pd
from starlette.testclient import TestClient

from api_prediction.main import app
from src.contracts import RunContext, SnapshotRef
from src.data_validation import validate_dataframe
from src.settings import settings


SAMPLE_SINGLE_GAME: Dict[str, Any] = {
    "appid": 1086940,
    "name": "Baldur's Gate 3",
    "price": 59.99,
    "required_age": 18,
    "release_date": "2023-08-03",
    "genres": ["RPG", "Strategy", "Adventure"],
    "windows": True,
    "mac": True,
    "linux": False,
}

SAMPLE_BATCH_GAMES: List[Dict[str, Any]] = [
    {
        "appid": 730,
        "name": "Counter-Strike 2",
        "price": 0.0,
        "required_age": 0,
        "release_date": "2012-08-21",
        "genres": ["Action", "Free to Play"],
        "windows": True,
        "mac": False,
        "linux": True,
    },
    {
        "appid": 1145360,
        "name": "Hades",
        "price": 24.99,
        "required_age": 0,
        "release_date": "2020-09-17",
        "genres": ["Action", "Indie", "RPG"],
        "windows": True,
        "mac": True,
        "linux": False,
    },
    {
        "appid": 1245620,
        "name": "Elden Ring",
        "price": 59.99,
        "required_age": 16,
        "release_date": "2022-02-25",
        "genres": ["Action", "RPG"],
        "windows": True,
        "mac": False,
        "linux": False,
    },
]


def step_1_doctor_check() -> Dict[str, Any]:
    """Étape 1 : Diagnostic de l'infrastructure et de l'environnement."""
    print("\n" + "=" * 80)
    print("  ÉTAPE 1 / 7 : DIAGNOSTIC D'ENVIRONNEMENT & SERVICES DOCKER")
    print("=" * 80)
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "doctor.py")]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    if res.returncode != 0:
        raise RuntimeError(f"Le diagnostic environnement a échoué :\n{res.stderr}")
    print(res.stdout)
    env_json_path = PROJECT_ROOT / "reports" / "environment.json"
    env_data = {}
    if env_json_path.exists():
        with open(env_json_path, "r", encoding="utf-8") as f:
            env_data = json.load(f)
    return {"status": "SUCCESS", "environment": env_data.get("status", "READY")}


def step_2_data_validation() -> Dict[str, Any]:
    """Étape 2 : Barrière de validation Great Expectations Core 1.x."""
    print("\n" + "=" * 80)
    print("  ÉTAPE 2 / 7 : VALIDATION DES DONNÉES & BARRIÈRE GREAT EXPECTATIONS")
    print("=" * 80)
    snapshot_path = PROJECT_ROOT / "data" / "snapshots" / "games_historical_v1.csv"
    if not snapshot_path.exists():
        # Fallback si absent
        alt_path = PROJECT_ROOT / "data" / "raw" / "games_march2025_full.csv"
        if alt_path.exists():
            snapshot_path = alt_path
        else:
            raise FileNotFoundError(f"Snapshot introuvable : {snapshot_path}")

    df = pd.read_csv(snapshot_path, nrows=5000)
    print(f"[DATA] Échantillon de contrôle chargé : {len(df)} jeux depuis {snapshot_path.name}")

    val_res = validate_dataframe(df, suite_name="historical_schema_v1")
    print(f"[GX] Barrière Great Expectations franchie avec succès :")
    print(f"  - Suite de validation : {val_res.get('suite_name')}")
    print(f"  - Succès global       : {val_res.get('success')}")
    print(f"  - Lignes auditées     : {val_res.get('row_count')}")
    print(f"  - Échecs constatés    : {val_res.get('failed_checks_count')}")

    if not val_res.get("success"):
        raise ValueError("La barrière Great Expectations a rejeté les données nominales !")

    return {
        "status": "SUCCESS",
        "suite": val_res.get("suite_name"),
        "row_count": val_res.get("row_count"),
        "failed_checks_count": val_res.get("failed_checks_count"),
        "success": val_res.get("success"),
    }


def step_3_orchestrated_pipeline(fast: bool = True) -> Dict[str, Any]:
    """Étape 3 : Entraînement orchestré Prefect & MLflow Tracking."""
    print("\n" + "=" * 80)
    print("  ÉTAPE 3 / 7 : PIPELINE ORCHESTRÉ PREFECT 3 & TRACKING MLFLOW")
    print("=" * 80)
    cmd = [
        sys.executable,
        "-m",
        "src.orchestration.tasks",
        "--batch-id",
        f"demo-{int(time.time())}",
    ]
    if fast:
        cmd.append("--fast")

    print(f"[PREFECT] Lancement du flow d'entraînement : {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    if res.returncode != 0:
        raise RuntimeError(f"Le pipeline orchestré a échoué :\n{res.stderr}\n{res.stdout}")

    print(res.stdout)
    return {"status": "SUCCESS", "mode": "fast" if fast else "full"}


def step_4_manifest_qualification() -> Dict[str, Any]:
    """Étape 4 : Qualification et intégrité cryptographique du manifeste."""
    print("\n" + "=" * 80)
    print("  ÉTAPE 4 / 7 : MANIFESTE DE SERVICE & INTÉGRITÉ CRYPTOGRAPHIQUE")
    print("=" * 80)
    manifest_path = PROJECT_ROOT / "state" / "service-manifest.json"

    # Synchronisation et consolidation du manifeste de service avec les modèles champions
    gen_cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "build_service_manifest.py")]
    subprocess.run(gen_cmd, check=True, cwd=str(PROJECT_ROOT))

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    manifest_sha = manifest_data.get("manifest_sha256")
    reg_ver = manifest_data.get("models", {}).get("regression", {}).get("version")
    clf_ver = manifest_data.get("models", {}).get("classification", {}).get("version")

    print(f"[MANIFEST] Manifeste de service actif : {manifest_data.get('manifest_id')}")
    print(f"  - Empreinte SHA-256   : {manifest_sha}")
    print(f"  - Version Régression   : v{reg_ver} ({manifest_data.get('models', {}).get('regression', {}).get('model_name')})")
    print(f"  - Version Classif.     : v{clf_ver} ({manifest_data.get('models', {}).get('classification', {}).get('model_name')})")
    print(f"  - Statut Champion      : {manifest_data.get('status')}")

    # Vérification de non-altération du manifeste
    from scripts.build_service_manifest import compute_manifest_sha256
    expected_sha = compute_manifest_sha256(manifest_data)
    if manifest_sha != expected_sha:
        raise ValueError(f"Altération du manifeste détectée : {manifest_sha} != {expected_sha}")

    print("  - Contrôle d'intégrité : Intégrité cryptographique validée sans altération.")
    return {
        "status": "SUCCESS",
        "manifest_id": manifest_data.get("manifest_id"),
        "manifest_sha256": manifest_sha,
        "regression_version": reg_ver,
        "classification_version": clf_ver,
    }


def step_5_api_predictions_and_leak_prevention(client: TestClient) -> Dict[str, Any]:
    """Étape 5 : Inférences unitaire, batch et barrière anti-fuite (HTTP 422)."""
    print("\n" + "=" * 80)
    print("  ÉTAPE 5 / 7 : INFÉRENCE API (UNITAIRE & BATCH) & CONTRÔLE ANTI-FUITE")
    print("=" * 80)

    results: Dict[str, Any] = {}

    # 1. Inférence unitaire
    resp_single = client.post("/predict", json={"game": SAMPLE_SINGLE_GAME})
    print(f"[API] POST /predict -> HTTP {resp_single.status_code}")
    if resp_single.status_code != 200:
        raise RuntimeError(f"Échec inférence unitaire : {resp_single.text}")
    data_single = resp_single.json()
    first_item = data_single.get("results", [])[0]
    preds = first_item.get("predictions", [])
    print(f"  - Request ID : {data_single.get('request_id')}")
    print(f"  - Pourcentage prédit : {first_item.get('predicted_pct')}%")
    print(f"  - Tranche estimée    : {first_item.get('predicted_owners')}")
    print(f"  - Durée traitement   : {data_single.get('processing_duration_ms'):.2f} ms")
    
    # Récupération de l'identifiant de prédiction pour le feedback
    reg_pred = next(p for p in preds if p.get("task") == "regression")
    results["single_prediction_id"] = reg_pred.get("prediction_id")

    # 2. Inférence par lot (batch vectorisé de 3 jeux)
    resp_batch = client.post("/predict/batch", json={"games": SAMPLE_BATCH_GAMES})
    print(f"[API] POST /predict/batch -> HTTP {resp_batch.status_code}")
    if resp_batch.status_code != 200:
        raise RuntimeError(f"Échec inférence batch : {resp_batch.text}")
    data_batch = resp_batch.json()
    batch_results = data_batch.get("results", [])
    total_preds = sum(len(item.get("predictions", [])) for item in batch_results)
    print(f"  - Jeux soumis : {data_batch.get('item_count')}")
    print(f"  - $2N$ Prédictions retournées : {total_preds}")
    print(f"  - Durée totale : {data_batch.get('processing_duration_ms'):.2f} ms")
    if total_preds != 6:
        raise ValueError(f"La cardinalité 2N n'est pas respectée (attendu 6, obtenu {total_preds}) !")

    # 3. Tentative de fuite de cible (doit être fermement rejetée en HTTP 422)
    leaked_payload = dict(SAMPLE_SINGLE_GAME)
    leaked_payload["pct_pos_total"] = 85.0
    resp_leak = client.post("/predict", json={"game": leaked_payload})
    print(f"[API] POST /predict (fuite injectée) -> HTTP {resp_leak.status_code}")
    if resp_leak.status_code != 422:
        raise ValueError(f"La fuite de cible n'a pas été rejetée en HTTP 422 (reçu: {resp_leak.status_code}) !")
    print("  - Barrière anti-fuite : Rejet formel HTTP 422 confirmé sans compromission.")

    results["status"] = "SUCCESS"
    results["batch_count"] = data_batch.get("item_count")
    results["leak_rejected_422"] = True
    return results


def step_6_feedback_and_idempotence(client: TestClient, prediction_id: str) -> Dict[str, Any]:
    """Étape 6 : Soumission de feedback et preuve d'idempotence (201 -> 200)."""
    print("\n" + "=" * 80)
    print("  ÉTAPE 6 / 7 : DÉPÔT DE RETOUR UTILISATEUR & IDEMPOTENCE DU FEEDBACK")
    print("=" * 80)

    fb_id = str(uuid4())
    now_iso = datetime.now(timezone.utc).isoformat()
    feedback_payload = {
        "feedback_id": fb_id,
        "prediction_id": prediction_id,
        "observed_pct": 92.5,
        "observed_at": now_iso,
        "label_source": "steam-store-actual",
        "useful": True,
    }

    # Premier envoi -> HTTP 201 Created
    resp1 = client.post("/feedback", json=feedback_payload)
    print(f"[FEEDBACK] Tentative 1 : POST /feedback -> HTTP {resp1.status_code}")
    if resp1.status_code != 201:
        raise RuntimeError(f"Échec création feedback initial : {resp1.text}")
    data1 = resp1.json()
    print(f"  - Feedback ID persisté : {data1.get('feedback_id')}")

    # Second envoi identique -> HTTP 200 OK (Idempotence)
    resp2 = client.post("/feedback", json=feedback_payload)
    print(f"[FEEDBACK] Tentative 2 : POST /feedback (identique) -> HTTP {resp2.status_code}")
    if resp2.status_code != 200:
        raise ValueError(f"L'idempotence a échoué (attendu 200, reçu {resp2.status_code}) !")
    print("  - Idempotence prouvée : Deuxième appel validé sans duplication ni conflit.")

    return {
        "status": "SUCCESS",
        "feedback_id": data1.get("feedback_id"),
        "idempotence_verified": True,
    }


def step_7_sql_audit_reconciliation() -> Dict[str, Any]:
    """Étape 7 : Rapprochement comptable SQL à l'unité près."""
    print("\n" + "=" * 80)
    print("  ÉTAPE 7 / 7 : AUDIT RELATIONNEL POSTGRESQL & RÉCONCILIATION SQL")
    print("=" * 80)
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "verify_sql_counts.py")]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    if res.returncode != 0:
        raise RuntimeError(f"La réconciliation SQL a échoué :\n{res.stderr}\n{res.stdout}")

    print(res.stdout)
    return {"status": "SUCCESS", "sql_reconciled": True}


def run_nominal_demo(campaign: str, fast: bool = True, with_incidents: bool = False) -> Dict[str, Any]:
    """Exécute la démonstration nominale complète."""
    start_time = time.time()
    t_iso = datetime.now(timezone.utc).isoformat()
    report: Dict[str, Any] = {
        "campaign": campaign,
        "started_at": t_iso,
        "fast_mode": fast,
        "with_incidents": with_incidents,
        "steps": {},
    }

    print("\n" + "#" * 80)
    print(f"   DÉMONSTRATION NOMINALE MLOPS 5MLDE")
    print(f"   Campagne : {campaign} | Horodatage : {t_iso}")
    print("#" * 80)

    try:
        # Étape 1
        report["steps"]["1_doctor"] = step_1_doctor_check()
        # Étape 2
        report["steps"]["2_data_validation"] = step_2_data_validation()
        # Étape 3
        report["steps"]["3_pipeline"] = step_3_orchestrated_pipeline(fast=fast)
        # Étape 4
        report["steps"]["4_manifest"] = step_4_manifest_qualification()

        # Étapes 5 & 6 : API Inférence et Feedback sous lifespan actif
        with TestClient(app) as client:
            pred_res = step_5_api_predictions_and_leak_prevention(client)
            report["steps"]["5_inference"] = pred_res
            report["steps"]["6_feedback"] = step_6_feedback_and_idempotence(client, pred_res["single_prediction_id"])

        # Étape 7
        report["steps"]["7_sql_audit"] = step_7_sql_audit_reconciliation()

        # Enchaînement optionnel des incidents
        if with_incidents:
            print("\n" + "=" * 80)
            print("  VOLET INCIDENTS, REPRISE & ROLLBACK (demo_incident.py)")
            print("=" * 80)
            inc_cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "demo_incident.py"),
                "--scenario",
                "all",
            ]
            res_inc = subprocess.run(inc_cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
            if res_inc.returncode != 0:
                raise RuntimeError(f"Le banc d'incidents a échoué :\n{res_inc.stderr}\n{res_inc.stdout}")
            print(res_inc.stdout)
            report["steps"]["incidents"] = {"status": "SUCCESS", "scenarios": 5}

        report["overall_status"] = "SUCCESS"
        elapsed = time.time() - start_time
        report["duration_seconds"] = round(elapsed, 2)

        print("\n" + "#" * 80)
        print(f"   ✅ DÉMONSTRATION NOMINALE VALIDÉE AVEC SUCCÈS ({elapsed:.2f} s)")
        print("#" * 80)

    except Exception as exc:
        report["overall_status"] = "FAILED"
        report["error"] = str(exc)
        logger.exception("Échec critique lors de la démonstration nominale")
        raise

    finally:
        # Sauvegarde du rapport d'exécution
        rep_path = PROJECT_ROOT / "reports" / "demo_nominal_report.json"
        with open(rep_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n[RAPPORT] Rapport de démonstration enregistré dans {rep_path.relative_to(PROJECT_ROOT)}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Démonstrateur nominal MLOps 5MLDE")
    parser.add_argument("--campaign", type=str, default="demo-nominal", help="Nom de la campagne de démonstration")
    parser.add_argument("--fast", action="store_true", default=True, help="Mode rapide pour l'entraînement (par défaut True)")
    parser.add_argument("--full-training", action="store_false", dest="fast", help="Entraînement complet (40 estimateurs)")
    parser.add_argument("--with-incidents", action="store_true", help="Enchaîne avec les 5 scénarios d'incidents")
    args = parser.parse_args()

    run_nominal_demo(
        campaign=args.campaign,
        fast=args.fast,
        with_incidents=args.with_incidents,
    )


if __name__ == "__main__":
    main()
