"""tests/integration/test_push_api.py - Tests d'intégration du service PUSH et des lots durables.

Vérifie :
1. L'idempotence des requêtes HTTP (200 sur duplicat exact, 409 sur conflit).
2. Le déclenchement transactionnel exact au seuil de 500 jeux uniques (scénario 499 / 500 / 501 / 1000).
3. L'étanchéité absolue et l'absence de chevauchement entre lots (disjointness).
4. La validation Pydantic bloquante (422 sur données invalides ou sentinelles).
5. L'exécution du worker et la résilience en cas d'échec de validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import pandas as pd
import pytest

from api_trigger_passif.main import create_app
from api_trigger_passif.storage import PushStorage
from api_trigger_passif.training_trigger import process_batch
from simulateur_push import generate_synthetic_push_event
from src.contracts import REQUIRED_SNAPSHOT_COLUMNS


@pytest.fixture
def push_client(tmp_path: Path):
    """Client de test FastAPI avec base SQLite isolée."""
    db_file = tmp_path / "test_push.db"
    app = create_app(db_path=db_file)
    client = TestClient(app)
    return client, app.state.storage, db_file


# ==============================================================================
# 1. Tests d'Idempotence & Contrôle des Conflits
# ==============================================================================

def test_push_idempotency_and_conflict(push_client):
    """Vérifie : 202 sur nouvelle soumission, 200 sur duplicat identique, 409 sur conflit."""
    client, storage, _ = push_client
    event = generate_synthetic_push_event(1)

    # 1. Première soumission -> 202 Accepted
    r1 = client.post("/games", json=event)
    assert r1.status_code == 202
    assert r1.json()["batch_created"] is False
    assert r1.json()["unassigned_count"] == 1

    # 2. Soumission strictement identique -> 200 OK (idempotent, aucun incrément)
    r2 = client.post("/games", json=event)
    assert r2.status_code == 200
    assert r2.json()["status"] == "idempotent_duplicate"

    status = client.get("/status").json()
    assert status["events_count"] == 1
    assert status["unassigned_games_count"] == 1

    # 3. Même event_id mais contenu différent -> 409 Conflict
    conflict_event = event.copy()
    conflict_event["game"] = event["game"].copy()
    conflict_event["game"]["name"] = "Conflicting Title"
    r3 = client.post("/games", json=conflict_event)
    assert r3.status_code == 409
    assert "Conflict" in r3.json()["error"]


def test_push_validation_error_on_invalid_payload(push_client):
    """Un payload invalide (tranche '0 - 0' ou prix négatif) doit retourner 422."""
    client, _, _ = push_client
    event = generate_synthetic_push_event(2)

    # Tranche '0 - 0' bannie pour les nouvelles observations
    event["game"]["estimated_owners"] = "0 - 0"
    r = client.post("/games", json=event)
    assert r.status_code == 422


# ==============================================================================
# 2. Scénarios de Seuil Strict (499, 500, 501, 1000)
# ==============================================================================

def test_threshold_triggers_batch_at_exact_500_games(push_client):
    """Démontre empiriquement le comportement sur les seuils 499, 500 et 501."""
    client, storage, _ = push_client

    # 1. Envoi de 499 événements -> aucun lot créé
    for i in range(1, 500):
        ev = generate_synthetic_push_event(i)
        r = client.post("/games", json=ev)
        assert r.status_code == 202

    status_499 = client.get("/status").json()
    assert status_499["events_count"] == 499
    assert status_499["unassigned_games_count"] == 499
    assert status_499["batches_count"] == 0

    # 2. Envoi du 500e événement -> déclenchement immédiat du lot 1
    ev_500 = generate_synthetic_push_event(500)
    r_500 = client.post("/games", json=ev_500)
    assert r_500.status_code == 202
    data_500 = r_500.json()
    assert data_500["batch_created"] is True
    batch_1_id = data_500["batch_id"]
    assert batch_1_id is not None
    assert data_500["unassigned_remaining"] == 0

    status_500 = client.get("/status").json()
    assert status_500["events_count"] == 500
    assert status_500["unassigned_games_count"] == 0
    assert status_500["batches_count"] == 1

    # 3. Envoi du 501e événement -> lot 1 reste isolé, 1 jeu en attente pour lot 2
    ev_501 = generate_synthetic_push_event(501)
    r_501 = client.post("/games", json=ev_501)
    assert r_501.status_code == 202
    assert r_501.json()["batch_created"] is False
    assert r_501.json()["unassigned_count"] == 1

    status_501 = client.get("/status").json()
    assert status_501["events_count"] == 501
    assert status_501["unassigned_games_count"] == 1
    assert status_501["batches_count"] == 1

    # 4. Vérification de la composition du lot 1
    batch_details = client.get(f"/batches/{batch_1_id}").json()
    assert batch_details["batch_id"] == batch_1_id
    assert batch_details["status"] == "PENDING"
    assert len(batch_details["appids"]) == 500


def test_batches_are_strictly_disjoint_at_1000_games(push_client):
    """Prouve qu'à 1000 jeux reçus, deux lots disjoints de 500 jeux sont constitués sans chevauchement."""
    client, storage, _ = push_client

    batch_ids = []
    # Envoi de 1000 jeux distincts
    for i in range(1, 1001):
        ev = generate_synthetic_push_event(i)
        r = client.post("/games", json=ev)
        assert r.status_code == 202
        if r.json().get("batch_created"):
            batch_ids.append(r.json()["batch_id"])

    assert len(batch_ids) == 2, "Exactement 2 lots doivent avoir été créés pour 1000 jeux."

    b1 = client.get(f"/batches/{batch_ids[0]}").json()
    b2 = client.get(f"/batches/{batch_ids[1]}").json()

    set_1 = set(b1["appids"])
    set_2 = set(b2["appids"])

    assert len(set_1) == 500
    assert len(set_2) == 500
    # Disjointness mathématique absolue
    assert set_1.isdisjoint(set_2), "Aucun jeu ne doit être partagé entre deux lots distincts !"

    status_1000 = client.get("/status").json()
    assert status_1000["unassigned_games_count"] == 0
    assert status_1000["batches_count"] == 2


# ==============================================================================
# 3. Traitement du Lot par le Worker et Résilience aux Échecs
# ==============================================================================

def test_process_batch_succeeds_and_updates_status(push_client, tmp_path: Path):
    """Vérifie le traitement complet d'un lot : extraction, validation GX push_batch_v1 et passage à SUCCEEDED."""
    client, storage, _ = push_client

    # Remplir un lot de 500 jeux
    batch_id = None
    for i in range(1, 501):
        ev = generate_synthetic_push_event(i)
        r = client.post("/games", json=ev)
        if r.json().get("batch_created"):
            batch_id = r.json()["batch_id"]

    assert batch_id is not None

    # Créer un snapshot de base factice
    base_rows = []
    for i in range(1, 1001):
        base_rows.append(
            {
                "appid": i,
                "name": f"Base Game {i}",
                "release_date": "2020-01-01",
                "required_age": 0,
                "price": 9.99,
                "dlc_count": 0,
                "achievements": 0,
                "windows": True,
                "mac": False,
                "linux": False,
                "positive": 100,
                "negative": 10,
                "num_reviews_total": 110,
                "pct_pos_total": 90.9,
                "genres": "['Action']",
                "estimated_owners": "20000 - 50000",
            }
        )
    base_file = tmp_path / "base.csv"
    pd.DataFrame(base_rows)[list(REQUIRED_SNAPSHOT_COLUMNS)].to_csv(base_file, index=False)

    # Traiter le lot
    res = process_batch(
        batch_id=batch_id,
        storage=storage,
        base_snapshot_path=base_file,
        snapshots_dir=tmp_path / "snapshots",
        trigger_training=False,
    )

    assert res["status"] == "SUCCEEDED"
    batch_record = client.get(f"/batches/{batch_id}").json()
    assert batch_record["status"] == "SUCCEEDED"
    assert batch_record["snapshot_path"] is not None
    assert batch_record["snapshot_sha256"] is not None


def test_process_batch_failure_retains_failed_state_without_crashing_api(push_client, tmp_path: Path):
    """Vérifie qu'un échec de validation marque le lot FAILED sans paralyser l'API."""
    client, storage, _ = push_client

    # Création d'un lot factice dans SQLite
    with storage._get_connection() as conn:
        conn.execute(
            "INSERT INTO batches (batch_id, status, created_at, row_count) VALUES (?, ?, ?, ?)",
            ("batch-corrupt", "PENDING", "2026-01-01T00:00:00Z", 0),
        )

    # Traitement du lot inexistant / corrompu -> FAILED
    res = process_batch(
        batch_id="batch-corrupt",
        storage=storage,
        snapshots_dir=tmp_path,
        trigger_training=False,
    )

    assert res["status"] == "FAILED"
    batch_record = client.get("/batches/batch-corrupt").json()
    assert batch_record["status"] == "FAILED"
    assert "Aucun jeu trouvé" in batch_record["failure_reason"]

    # L'API reste 100% fonctionnelle et accepte de nouveaux événements
    health = client.get("/health").json()
    assert health["status"] == "healthy"
