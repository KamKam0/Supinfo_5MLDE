"""scripts/verify_sql_counts.py - Réconciliation des métriques SQL sur une campagne connue.

Exécute et réconcilie la campagne étalonnée d'audit :
- 2 demandes unitaires
- 1 lot de 3 jeux
- 1 entrée invalide (fuite de cible 422)
- 2 créations de feedbacks (201)
- 1 rejeu identique de feedback (200)

Total : exactement 7 tentatives métier, 5 jeux prédits, 10 prédictions, 2 feedbacks persistés.
Vérifie à l'unité près les données brutes SQL de tracing.api_requests, tracing.predictions, tracing.feedbacks.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import psycopg
from starlette.testclient import TestClient

from api_prediction.main import app
from src.settings import settings


SAMPLE_GAMES: List[Dict[str, Any]] = [
    {
        "appid": 1086940,
        "name": "Baldur's Gate 3",
        "price": 59.99,
        "required_age": 18,
        "release_date": "2023-08-03",
        "genres": ["RPG", "Strategy", "Adventure"],
        "windows": True,
        "mac": True,
        "linux": False,
    },
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
]


def execute_known_campaign(campaign_id: str, client: TestClient) -> Dict[str, Any]:
    """Déclenche la séquence exacte de 7 tentatives métier sous l'en-tête X-Campaign-ID."""
    print(f"\n[CAMPAIGN] Déclenchement de la campagne d'audit '{campaign_id}'...")
    results = {}

    headers = {"X-Campaign-ID": campaign_id}

    # 1. Première demande unitaire
    r1 = client.post("/predict", json={"game": SAMPLE_GAMES[0]}, headers=headers)
    assert r1.status_code == 200, f"Req 1 failed: {r1.status_code} - {r1.text}"
    p1_data = r1.json()
    pred_reg_id_1 = next(
        p["prediction_id"] for p in p1_data["results"][0]["predictions"] if p["task"] == "regression"
    )

    # 2. Deuxième demande unitaire
    r2 = client.post("/predict", json={"game": SAMPLE_GAMES[1]}, headers=headers)
    assert r2.status_code == 200, f"Req 2 failed: {r2.status_code} - {r2.text}"
    p2_data = r2.json()
    pred_cls_id_2 = next(
        p["prediction_id"] for p in p2_data["results"][0]["predictions"] if p["task"] == "classification"
    )

    # 3. Lot de 3 jeux (vectorisé)
    r3 = client.post("/predict/batch", json={"games": SAMPLE_GAMES}, headers=headers)
    assert r3.status_code == 200, f"Req 3 failed: {r3.status_code}"

    # 4. Entrée invalide avec fuite de cible (doit être rejetée 422)
    leaky_game = dict(SAMPLE_GAMES[0])
    leaky_game["pct_pos_total"] = 85.5  # Target leakage interdit !
    r4 = client.post("/predict", json={"game": leaky_game}, headers=headers)
    assert r4.status_code == 422, f"Req 4 non rejetée: {r4.status_code}"

    # 5. Premier feedback conforme (201 Created)
    fb1_id = str(uuid.uuid4())
    fb1_payload = {
        "feedback_id": fb1_id,
        "prediction_id": pred_reg_id_1,
        "observed_pct": 92.5,
        "observed_at": "2026-11-16T15:00:00Z",
        "label_source": "steam-store-actual",
        "useful": True,
    }
    r5 = client.post("/feedback", json=fb1_payload, headers=headers)
    assert r5.status_code == 201, f"Req 5 feedback 1 failed: {r5.status_code} - {r5.text}"

    # 6. Deuxième feedback conforme (201 Created)
    fb2_id = str(uuid.uuid4())
    fb2_payload = {
        "feedback_id": fb2_id,
        "prediction_id": pred_cls_id_2,
        "observed_owners": "50000000 - 100000000",
        "observed_at": "2026-11-16T15:00:00Z",
        "label_source": "steam-store-actual",
        "useful": True,
    }
    r6 = client.post("/feedback", json=fb2_payload, headers=headers)
    assert r6.status_code == 201, f"Req 6 feedback 2 failed: {r6.status_code} - {r6.text}"

    # 7. Rejeu identique du premier feedback (200 OK idempotent)
    r7 = client.post("/feedback", json=fb1_payload, headers=headers)
    assert r7.status_code == 200, f"Req 7 feedback replay failed: {r7.status_code} - {r7.text}"

    print("[CAMPAIGN] 7 tentatives métier exécutées avec succès.")
    return {
        "campaign_id": campaign_id,
        "pred_reg_id_1": pred_reg_id_1,
        "pred_cls_id_2": pred_cls_id_2,
        "fb1_id": fb1_id,
        "fb2_id": fb2_id,
    }


def verify_sql_reconciliation(campaign_id: str) -> bool:
    """Interroge directement PostgreSQL et compare les totaux observés aux attentes strictes."""
    print(f"\n[RECONCILIATION] Rapprochement SQL des totaux pour la campagne '{campaign_id}'...")
    conn_url = f"postgresql://grafana_reader:5mlde-local-grafana@127.0.0.1:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"

    with psycopg.connect(conn_url) as conn:
        with conn.cursor() as cur:
            # 1. Total des requêtes enregistrées
            cur.execute(
                """
                SELECT
                    count(*) AS total_attempts,
                    count(*) FILTER (WHERE http_status = 200) AS count_200,
                    count(*) FILTER (WHERE http_status = 201) AS count_201,
                    count(*) FILTER (WHERE http_status = 422) AS count_422,
                    count(*) FILTER (WHERE http_status >= 500) AS count_5xx,
                    COALESCE(sum(item_count) FILTER (WHERE http_status = 200 AND endpoint IN ('/predict', '/predict/batch')), 0) AS total_games
                FROM tracing.api_requests
                WHERE campaign_id = %s;
                """,
                (campaign_id,),
            )
            req_stats = cur.fetchone()

            # 2. Total des prédictions générées
            cur.execute(
                """
                SELECT
                    count(p.prediction_id) AS total_predictions,
                    count(p.prediction_id) FILTER (WHERE p.task = 'regression') AS count_regression,
                    count(p.prediction_id) FILTER (WHERE p.task = 'classification') AS count_classification
                FROM tracing.predictions p
                JOIN tracing.api_requests r ON r.request_id = p.request_id
                WHERE r.campaign_id = %s;
                """,
                (campaign_id,),
            )
            pred_stats = cur.fetchone()

            # 3. Total des feedbacks persistés
            cur.execute(
                """
                SELECT count(f.feedback_id) AS total_feedbacks
                FROM tracing.feedbacks f
                JOIN tracing.predictions p ON p.prediction_id = f.prediction_id
                JOIN tracing.api_requests r ON r.request_id = p.request_id
                WHERE r.campaign_id = %s;
                """,
                (campaign_id,),
            )
            fb_stats = cur.fetchone()

            # 4. Contrôle cardinalité stricte (2N par jeu)
            cur.execute(
                """
                SELECT
                    r.request_id,
                    r.item_count,
                    count(p.prediction_id) AS prediction_count,
                    count(DISTINCT p.item_index) AS observed_items
                FROM tracing.api_requests r
                LEFT JOIN tracing.predictions p ON p.request_id = r.request_id
                WHERE r.endpoint IN ('/predict', '/predict/batch')
                  AND r.http_status = 200
                  AND r.campaign_id = %s
                GROUP BY r.request_id, r.item_count
                HAVING count(p.prediction_id) <> 2 * r.item_count
                    OR count(DISTINCT p.item_index) <> r.item_count;
                """,
                (campaign_id,),
            )
            cardinality_violations = cur.fetchall()

    total_attempts, count_200, count_201, count_422, count_5xx, total_games = req_stats
    total_preds, count_reg, count_cls = pred_stats
    total_feedbacks = fb_stats[0]

    expectations = [
        ("Tentatives HTTP totales (api_requests)", 7, total_attempts),
        ("Requêtes HTTP 200 (succès prédictions & replay)", 4, count_200),  # 2 predict + 1 batch + 1 replay fb
        ("Requêtes HTTP 201 (création nouveaux feedbacks)", 2, count_201),  # 2 créations feedback
        ("Requêtes HTTP 422 (rejet fuite de cible)", 1, count_422),
        ("Erreurs HTTP 5xx (indisponibilité)", 0, count_5xx),
        ("Jeux prédits avec succès (somme item_count)", 5, total_games),
        ("Lignes de prédiction totales (predictions)", 10, total_preds),
        ("Prédictions tâche régression (pct_pos)", 5, count_reg),
        ("Prédictions tâche classification (owners)", 5, count_cls),
        ("Feedbacks persistés uniques (feedbacks)", 2, total_feedbacks),
        ("Violations de cardinalité 2N (requêtes anormales)", 0, len(cardinality_violations)),
    ]

    all_passed = True
    print("\n" + "=" * 80)
    print(f"{'MÉTRIQUE CONTRÔLÉE':<50} | {'ATTENDU':<10} | {'OBSERVÉ SQL':<10} | {'STATUT':<8}")
    print("-" * 80)
    for name, expected, observed in expectations:
        match = expected == observed
        status = "OK" if match else "MISMATCH"
        if not match:
            all_passed = False
        print(f"{name:<50} | {expected:<10} | {observed:<10} | {status:<8}")
    print("=" * 80)

    if all_passed:
        print("\n[RÉCONCILIATION RÉUSSIE] 100% de concordance exacte à l'unité près !")
    else:
        print("\n[RÉCONCILIATION ÉCHOUÉE] Des écarts ont été constatés entre les totaux attendus et PostgreSQL.", file=sys.stderr)

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Script de réconciliation SQL des comptages d'audit")
    parser.add_argument("--campaign", type=str, default="demo-nominal", help="Nom de la campagne d'audit")
    parser.add_argument("--run-campaign", action="store_true", default=True, help="Déclenche la séquence de requêtes avant réconciliation")
    args = parser.parse_args()

    # Initialisation de l'API de prédiction
    with TestClient(app) as client:
        if args.run_campaign:
            # Génère un identifiant unique par exécution si demo-nominal
            campaign_tag = f"{args.campaign}-{uuid.uuid4().hex[:6]}"
            execute_known_campaign(campaign_tag, client)
        else:
            campaign_tag = args.campaign

        success = verify_sql_reconciliation(campaign_tag)
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
