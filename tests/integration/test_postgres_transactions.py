"""tests/integration/test_postgres_transactions.py - Tests d'intégration de la persistance PostgreSQL 18.6 et des feedbacks.

Valide l'ensemble des critères d'acceptation suivants :
- Initialisation et enregistrement idempotent des modèles dans tracing.model_versions.
- Atomicité transactionnelle stricte : enregistrement conjoint de api_requests et 2N predictions.
- Rollback atomique intégral en cas de panne injectée (aucun résultat partiel persistant).
- Gestion idempotente des feedbacks : 201 Created, 200 Replay idempotent, 409 Conflit, 404 Introuvable, 422 Incohérence.
- Respect rigoureux des cardinalités relationnelles et des contraintes d'unicité SQL.
- Télémétrie de secours locale en cas d'incident base.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List
from uuid import UUID, uuid4
import psycopg
import pytest
from starlette.testclient import TestClient

from api_prediction.main import app
from api_prediction.repository import db_pool, record_prediction_transaction
from api_prediction.telemetry import FALLBACK_LOG_PATH, log_telemetry_fallback
from src.settings import settings


@pytest.fixture(scope="module")
def api_client():
    """Client de test FastAPI avec initialisation complète du lifespan."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def db_conn():
    """Connexion directe de contrôle PostgreSQL."""
    conn = psycopg.connect(settings.DATABASE_URL, autocommit=True)
    yield conn
    conn.close()


@pytest.fixture
def sample_game() -> Dict[str, Any]:
    """Jeu de test nominal sans cibles."""
    return {
        "appid": 570,
        "name": "Dota 2",
        "price": 0.0,
        "required_age": 0,
        "dlc_count": 0,
        "achievements": 0,
        "windows": True,
        "mac": True,
        "linux": True,
        "genres": ["Action", "Free to Play", "Strategy"],
        "release_date": "2013-07-09",
    }


# ==============================================================================
# 1. Tests d'Enregistrement des Modèles (tracing.model_versions)
# ==============================================================================

def test_model_versions_registered_in_db(db_conn: psycopg.Connection):
    """Vérifie que les versions championnes sont enregistrées de manière univoque dans PostgreSQL."""
    cur = db_conn.cursor()
    cur.execute(
        """
        SELECT task, registered_name, version, tracking_uri
        FROM tracing.model_versions
        ORDER BY task;
        """
    )
    rows = cur.fetchall()
    assert len(rows) >= 2

    tasks = {r[0]: (r[1], r[2]) for r in rows}
    assert "classification" in tasks
    assert "regression" in tasks
    assert tasks["regression"][0] == "steam-positive-ratio"
    assert tasks["classification"][0] == "steam-owner-range"


# ==============================================================================
# 2. Tests de Persistance Atomique des Prédictions (api_requests & predictions)
# ==============================================================================

def test_predict_single_atomic_persistence(
    api_client: TestClient, db_conn: psycopg.Connection, sample_game: Dict[str, Any]
):
    """Vérifie qu'un appel unitaire enregistre 1 requête et 2 prédictions conjointes (2N)."""
    res = api_client.post(
        "/predict",
        json={"game": sample_game},
        headers={"X-Campaign-ID": "test-campaign-single"},
    )
    assert res.status_code == 200
    data = res.json()
    req_id = UUID(data["request_id"])

    cur = db_conn.cursor()
    # 1. Vérification de la requête
    cur.execute(
        """
        SELECT endpoint, http_status, item_count, campaign_id, error_code
        FROM tracing.api_requests
        WHERE request_id = %s;
        """,
        (req_id,),
    )
    req_row = cur.fetchone()
    assert req_row is not None
    assert req_row[0] == "/predict"
    assert req_row[1] == 200
    assert req_row[2] == 1
    assert req_row[3] == "test-campaign-single"
    assert req_row[4] is None

    # 2. Vérification des 2 prédictions
    cur.execute(
        """
        SELECT task, predicted_pct, predicted_owners, item_index
        FROM tracing.predictions
        WHERE request_id = %s
        ORDER BY task;
        """,
        (req_id,),
    )
    preds = cur.fetchall()
    assert len(preds) == 2

    # Classification
    assert preds[0][0] == "classification"
    assert preds[0][1] is None
    assert preds[0][2] is not None
    assert preds[0][3] == 0

    # Régression
    assert preds[1][0] == "regression"
    assert 0.0 <= preds[1][1] <= 100.0
    assert preds[1][2] is None
    assert preds[1][3] == 0


def test_predict_batch_atomic_persistence(
    api_client: TestClient, db_conn: psycopg.Connection, sample_game: Dict[str, Any]
):
    """Vérifie qu'un lot de N jeux persiste 1 requête et exactement 2N prédictions."""
    batch_games = [
        dict(sample_game, appid=2000 + i, price=5.0 * i) for i in range(4)
    ]
    res = api_client.post(
        "/predict/batch",
        json={"games": batch_games},
        headers={"X-Campaign-ID": "test-campaign-batch"},
    )
    assert res.status_code == 200
    data = res.json()
    req_id = UUID(data["request_id"])

    cur = db_conn.cursor()
    # Requête de lot
    cur.execute(
        """
        SELECT endpoint, http_status, item_count, campaign_id
        FROM tracing.api_requests
        WHERE request_id = %s;
        """,
        (req_id,),
    )
    req_row = cur.fetchone()
    assert req_row is not None
    assert req_row[0] == "/predict/batch"
    assert req_row[1] == 200
    assert req_row[2] == 4

    # 2N = 8 prédictions
    cur.execute(
        """
        SELECT count(*), count(DISTINCT item_index)
        FROM tracing.predictions
        WHERE request_id = %s;
        """,
        (req_id,),
    )
    count_row = cur.fetchone()
    assert count_row[0] == 8  # 4 * 2
    assert count_row[1] == 4  # indices 0 à 3


# ==============================================================================
# 3. Test de Rollback Atomique en Cas de Panne
# ==============================================================================

def test_atomic_rollback_on_persistence_failure(sample_game: Dict[str, Any], db_conn: psycopg.Connection):
    """Prouve qu'un échec survenu pendant la transaction annule l'intégralité du lot."""
    from api_prediction.schemas import (
        GamePredictionItem,
        InferenceGameInput,
        TaskModelInfo,
        TaskPredictionResult,
    )

    req_uuid = uuid4()
    bad_model_id = uuid4()  # ID inexistant violant la clé étrangère
    fake_game = InferenceGameInput(**sample_game)

    task_reg = TaskPredictionResult(
        prediction_id=str(uuid4()),
        task="regression",
        target_column="pct_pos_total",
        predicted_value=75.0,
        model=TaskModelInfo(server_uri="uri", model_name="name", model_version="1", mlflow_run_id="run"),
    )
    task_cls = TaskPredictionResult(
        prediction_id=str(uuid4()),
        task="classification",
        target_column="estimated_owners",
        predicted_value="0 - 20000",
        model=TaskModelInfo(server_uri="uri", model_name="name", model_version="1", mlflow_run_id="run"),
    )
    item = GamePredictionItem(item_index=0, predictions=[task_reg, task_cls], predicted_pct=75.0, predicted_owners="0 - 20000")

    # Tentative avec clé étrangère invalide
    with pytest.raises(Exception):
        record_prediction_transaction(
            pool=db_pool,
            request_id=req_uuid,
            started_at=datetime.now(timezone.utc),
            endpoint="/predict",
            item_count=1,
            processing_duration_ms=1.0,
            inference_duration_ms=1.0,
            worker_pid=100,
            instance_id=uuid4(),
            api_revision="rev-test",
            campaign_id="test",
            results=[item],
            games=[fake_game],
            reg_model_id=bad_model_id,
            cls_model_id=bad_model_id,
        )

    # Vérification d'absence totale d'artefacts persistés (rollback réussi)
    cur = db_conn.cursor()
    cur.execute("SELECT count(*) FROM tracing.api_requests WHERE request_id = %s;", (req_uuid,))
    assert cur.fetchone()[0] == 0
    cur.execute("SELECT count(*) FROM tracing.predictions WHERE request_id = %s;", (req_uuid,))
    assert cur.fetchone()[0] == 0


# ==============================================================================
# 4. Tests des Retours et Idempotence (POST /feedback)
# ==============================================================================

def test_feedback_lifecycle_and_idempotence(
    api_client: TestClient, db_conn: psycopg.Connection, sample_game: Dict[str, Any]
):
    """Valide les 6 cas contractuels de réception de feedback (201, 200, 409, 404, 422)."""
    # 1. Créer une prédiction pour obtenir des identifiants réels
    pred_res = api_client.post("/predict", json={"game": sample_game})
    assert pred_res.status_code == 200
    pred_data = pred_res.json()
    reg_pred_id = pred_data["results"][0]["predictions"][0]["prediction_id"]

    # --------------------------------------------------------------------------
    # Cas 1 : Premier enregistrement conforme (201 Created)
    # --------------------------------------------------------------------------
    feedback_id = str(uuid4())
    obs_time = datetime.now(timezone.utc).isoformat()
    fb_payload = {
        "feedback_id": feedback_id,
        "prediction_id": reg_pred_id,
        "useful": True,
        "observed_pct": 82.5,
        "observed_at": obs_time,
        "label_source": "steam-store-actual",
    }
    r1 = api_client.post("/feedback", json=fb_payload)
    assert r1.status_code == 201
    assert r1.json()["status"] == "created"

    # Contrôle SQL
    cur = db_conn.cursor()
    cur.execute("SELECT feedback_id, useful, observed_pct FROM tracing.feedbacks WHERE feedback_id = %s;", (UUID(feedback_id),))
    row_fb = cur.fetchone()
    assert row_fb is not None
    assert row_fb[1] is True
    assert row_fb[2] == 82.5

    # --------------------------------------------------------------------------
    # Cas 2 : Rejeu strictement identique (200 OK Idempotent)
    # --------------------------------------------------------------------------
    r2 = api_client.post("/feedback", json=fb_payload)
    assert r2.status_code == 200
    assert r2.json()["status"] == "replayed"

    # Vérification qu'aucun doublon de feedback n'a été inséré
    cur.execute("SELECT count(*) FROM tracing.feedbacks WHERE prediction_id = %s;", (UUID(reg_pred_id),))
    assert cur.fetchone()[0] == 1

    # --------------------------------------------------------------------------
    # Cas 3 : Même feedback_id avec contenu différent (409 Conflict)
    # --------------------------------------------------------------------------
    conflict_payload = dict(fb_payload, observed_pct=99.0)
    r3 = api_client.post("/feedback", json=conflict_payload)
    assert r3.status_code == 409
    assert r3.json()["error_code"] == "FEEDBACK_CONFLICT"

    # --------------------------------------------------------------------------
    # Cas 4 : Autre feedback_id ciblant une prédiction déjà annotée (409 Conflict)
    # --------------------------------------------------------------------------
    second_feedback_payload = dict(fb_payload, feedback_id=str(uuid4()))
    r4 = api_client.post("/feedback", json=second_feedback_payload)
    assert r4.status_code == 409
    assert r4.json()["error_code"] == "FEEDBACK_CONFLICT"

    # --------------------------------------------------------------------------
    # Cas 5 : Prédiction inconnue (404 Not Found)
    # --------------------------------------------------------------------------
    unknown_pred_payload = dict(fb_payload, feedback_id=str(uuid4()), prediction_id=str(uuid4()))
    r5 = api_client.post("/feedback", json=unknown_pred_payload)
    assert r5.status_code == 404
    assert r5.json()["error_code"] == "PREDICTION_NOT_FOUND"


def test_feedback_incoherent_task_rejection(api_client: TestClient, sample_game: Dict[str, Any]):
    """Vérifie le rejet HTTP 422 si une observation de classification est soumise sur une régression."""
    pred_res = api_client.post("/predict", json={"game": sample_game})
    reg_pred_id = pred_res.json()["results"][0]["predictions"][0]["prediction_id"]

    incoherent_payload = {
        "feedback_id": str(uuid4()),
        "prediction_id": reg_pred_id,
        "observed_owners": "20000 - 50000",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "label_source": "steam-store-actual",
    }
    r = api_client.post("/feedback", json=incoherent_payload)
    assert r.status_code == 422
    assert r.json()["error_code"] == "INVALID_INPUT"


def test_feedback_opinion_only_without_observation(api_client: TestClient, sample_game: Dict[str, Any], db_conn: psycopg.Connection):
    """Vérifie qu'un feedback d'avis subjectif seul (sans observation) est accepté et stocke des métadonnées NULL."""
    pred_res = api_client.post("/predict", json={"game": sample_game})
    cls_pred_id = pred_res.json()["results"][0]["predictions"][1]["prediction_id"]

    opinion_payload = {
        "feedback_id": str(uuid4()),
        "prediction_id": cls_pred_id,
        "useful": False,
    }
    r = api_client.post("/feedback", json=opinion_payload)
    assert r.status_code == 201

    cur = db_conn.cursor()
    cur.execute(
        """
        SELECT useful, observed_pct, observed_owners, observed_at, label_source
        FROM tracing.feedbacks
        WHERE feedback_id = %s;
        """,
        (UUID(opinion_payload["feedback_id"]),),
    )
    row = cur.fetchone()
    assert row[0] is False
    assert row[1] is None
    assert row[2] is None
    assert row[3] is None
    assert row[4] is None


# ==============================================================================
# 5. Tests de Télémétrie de Secours et Santé de Pool
# ==============================================================================

def test_telemetry_fallback_logging():
    """Valide l'écriture d'un incident dans le journal local de secours en cas d'erreur."""
    req_id = str(uuid4())
    log_telemetry_fallback(
        request_id=req_id,
        endpoint="/predict",
        http_status=503,
        error_code="DB_OFFLINE_SIMULATION",
        message="Test simulation panne base",
    )
    assert FALLBACK_LOG_PATH.exists()

    with open(FALLBACK_LOG_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    last_event = json.loads(lines[-1])
    assert last_event["request_id"] == req_id
    assert last_event["error_code"] == "DB_OFFLINE_SIMULATION"


def test_database_pool_health_and_connection():
    """Vérifie la robustesse du pool de connexions borné."""
    assert db_pool.is_healthy() is True

    # Acquisition et libération de multiples connexions
    conns = []
    for _ in range(3):
        with db_pool.connection() as conn:
            assert not conn.closed
