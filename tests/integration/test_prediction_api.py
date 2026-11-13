"""tests/integration/test_prediction_api.py - Tests d'intégration de l'API d'Inférence FastAPI (Port 8001).

Vérifie l'ensemble des critères d'acceptation de l'API d'inférence :
- Démarrage conforme du cycle de vie lifespan et chargement unique en mémoire des modèles.
- Sonde live (HTTP 200) et sonde ready (HTTP 200 / HTTP 503 en cas d'indisponibilité).
- Inférence unitaire /predict (formats enveloppés 'game' et direct).
- Inférence vectorisée par lot /predict/batch (1 à 100 jeux).
- Équivalence mathématique stricte entre lot vectorisé et N appels unitaires.
- Barrière anti-fuite absolue : rejet HTTP 422 sur toute variable cible (pct_pos_total, estimated_owners, etc.).
- Rejet des lots vides ou excédant 100 éléments.
- Corrélation HTTP : propagation universelle de X-Request-ID et en-têtes d'audit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import pytest
from starlette.testclient import TestClient

from api_prediction.main import app
from src.settings import settings


@pytest.fixture(scope="module")
def api_client():
    """Client de test FastAPI avec déclenchement complet du lifespan."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def valid_single_game() -> Dict[str, Any]:
    """Exemple de jeu conforme sans aucune variable cible."""
    return {
        "appid": 1086940,
        "name": "Baldur's Gate 3",
        "price": 59.99,
        "required_age": 18,
        "dlc_count": 2,
        "achievements": 53,
        "windows": True,
        "mac": True,
        "linux": False,
        "genres": ["RPG", "Adventure", "Strategy"],
        "release_date": "2023-08-03",
    }


@pytest.fixture
def sample_batch_games() -> List[Dict[str, Any]]:
    """Échantillon de jeux hétérogènes pour le test de lots."""
    return [
        {
            "appid": 1001,
            "name": "Indie Platformer",
            "price": 4.99,
            "required_age": 0,
            "dlc_count": 0,
            "achievements": 12,
            "windows": True,
            "mac": False,
            "linux": False,
            "genres": ["Indie", "Action"],
            "release_date": "2021-06-15",
        },
        {
            "appid": 1002,
            "name": "Strategy Commander",
            "price": 29.99,
            "required_age": 12,
            "dlc_count": 5,
            "achievements": 40,
            "windows": True,
            "mac": True,
            "linux": True,
            "genres": ["Strategy", "Simulation"],
            "release_date": "2022-11-20",
        },
        {
            "appid": 1003,
            "name": "Free Casual Puzzle",
            "price": 0.0,
            "required_age": 0,
            "dlc_count": 0,
            "achievements": 100,
            "windows": True,
            "mac": False,
            "linux": False,
            "genres": ["Casual", "Free to Play"],
            "release_date": "2023-02-10",
        },
    ]


# ==============================================================================
# 1. Tests des Sondes de Santé et Métadonnées
# ==============================================================================

def test_health_live_endpoint(api_client: TestClient):
    """Vérifie la vivacité du processus API."""
    res = api_client.get("/health/live")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "live"
    assert data["service"] == "steam-prediction-api"
    assert "timestamp" in data
    assert "X-Request-ID" in res.headers


def test_health_ready_endpoint(api_client: TestClient):
    """Vérifie l'aptitude opérationnelle et le chargement préalable des modèles."""
    res = api_client.get("/health/ready")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ready"
    assert data["models_loaded"] is True
    assert data["instance_id"] is not None
    assert data["manifest_id"] is not None
    assert res.headers["X-Manifest-ID"] == data["manifest_id"]


def test_models_info_endpoint(api_client: TestClient):
    """Vérifie la consultation du manifeste actif et de ses empreintes."""
    res = api_client.get("/models")
    assert res.status_code == 200
    data = res.json()
    assert "manifest_id" in data
    assert "manifest_sha256" in data
    assert "models" in data
    assert "regression" in data["models"]
    assert "classification" in data["models"]
    assert data["models"]["regression"]["model_name"] == "steam-positive-ratio"
    assert data["models"]["classification"]["model_name"] == "steam-owner-range"


# ==============================================================================
# 2. Tests d'Inférence Unitaire
# ==============================================================================

def test_predict_single_wrapped_format(api_client: TestClient, valid_single_game: Dict[str, Any]):
    """Vérifie l'inférence unitaire avec le format standard '{"game": {...}}'."""
    payload = {"game": valid_single_game}
    res = api_client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["item_count"] == 1
    assert len(data["results"]) == 1
    item = data["results"][0]
    assert item["item_index"] == 0
    assert item["appid"] == valid_single_game["appid"]

    # Vérification des deux prédictions conjointes
    assert len(item["predictions"]) == 2
    tasks = {p["task"]: p for p in item["predictions"]}
    assert "regression" in tasks
    assert "classification" in tasks

    # Régression : pct_pos_total dans [0, 100]
    reg_pred = tasks["regression"]
    assert reg_pred["target_column"] == "pct_pos_total"
    assert 0.0 <= float(reg_pred["predicted_value"]) <= 100.0
    assert item["predicted_pct"] == reg_pred["predicted_value"]

    # Classification : tranche non vide
    cls_pred = tasks["classification"]
    assert cls_pred["target_column"] == "estimated_owners"
    assert isinstance(cls_pred["predicted_value"], str) and len(cls_pred["predicted_value"]) > 0
    assert item["predicted_owners"] == cls_pred["predicted_value"]

    # Vérification des durées et en-têtes
    assert data["processing_duration_ms"] >= 0.0
    assert data["inference_duration_ms"] >= 0.0
    assert res.headers["X-Request-ID"] == data["request_id"]
    assert "X-Model-Regression-Version" in res.headers
    assert "X-Model-Classification-Version" in res.headers


def test_predict_single_direct_body_format(api_client: TestClient, valid_single_game: Dict[str, Any]):
    """Vérifie l'inférence unitaire acceptant le corps direct du jeu '{...}'."""
    res = api_client.post("/predict", json=valid_single_game)
    assert res.status_code == 200
    data = res.json()
    assert data["item_count"] == 1
    assert data["results"][0]["appid"] == valid_single_game["appid"]


# ==============================================================================
# 3. Tests d'Inférence par Lot (Vectorisée)
# ==============================================================================

def test_predict_batch_success(api_client: TestClient, sample_batch_games: List[Dict[str, Any]]):
    """Vérifie l'inférence vectorisée par lot avec la clé 'games'."""
    payload = {"games": sample_batch_games}
    res = api_client.post("/predict/batch", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["item_count"] == 3
    assert len(data["results"]) == 3

    for idx, item in enumerate(data["results"]):
        assert item["item_index"] == idx
        assert item["appid"] == sample_batch_games[idx]["appid"]
        assert len(item["predictions"]) == 2
        assert 0.0 <= item["predicted_pct"] <= 100.0
        assert item["predicted_owners"] is not None


def test_predict_batch_items_alias(api_client: TestClient, sample_batch_games: List[Dict[str, Any]]):
    """Vérifie le support de l'alias 'items' pour le lot."""
    payload = {"items": sample_batch_games}
    res = api_client.post("/predict/batch", json=payload)
    assert res.status_code == 200
    assert res.json()["item_count"] == 3


# ==============================================================================
# 4. Équivalence Mathématique Stricte (Lot Vectorisé vs N Appels Unitaires)
# ==============================================================================

def test_batch_equivalence_with_single_calls(
    api_client: TestClient, sample_batch_games: List[Dict[str, Any]]
):
    """Prouve que l'inférence vectorisée produit des résultats rigoureusement identiques à N appels unitaires."""
    # 1. Inférence vectorisée
    batch_res = api_client.post("/predict/batch", json={"games": sample_batch_games})
    assert batch_res.status_code == 200
    batch_data = batch_res.json()

    # 2. Inférences unitaires séquentielles
    single_results = []
    for g in sample_batch_games:
        s_res = api_client.post("/predict", json={"game": g})
        assert s_res.status_code == 200
        single_results.append(s_res.json()["results"][0])

    # 3. Comparaison rigoureuse
    for idx in range(len(sample_batch_games)):
        b_item = batch_data["results"][idx]
        s_item = single_results[idx]

        assert b_item["appid"] == s_item["appid"]
        # Équivalence numérique sur le pourcentage
        assert abs(b_item["predicted_pct"] - s_item["predicted_pct"]) < 1e-4, (
            f"Écart de régression pour l'élément {idx}: batch={b_item['predicted_pct']} != single={s_item['predicted_pct']}"
        )
        # Équivalence stricte sur la tranche
        assert b_item["predicted_owners"] == s_item["predicted_owners"], (
            f"Écart de classification pour l'élément {idx}: batch={b_item['predicted_owners']} != single={s_item['predicted_owners']}"
        )


# ==============================================================================
# 5. Barrière Anti-Fuite Absolue (Rejet 422 - REQ-16)
# ==============================================================================

@pytest.mark.parametrize(
    "forbidden_field,value",
    [
        ("pct_pos_total", 85.5),
        ("estimated_owners", "20,000 .. 50,000"),
        ("num_reviews_total", 1250),
        ("positive", 1100),
        ("negative", 150),
    ],
)
def test_rejection_target_leakage_single(
    api_client: TestClient, valid_single_game: Dict[str, Any], forbidden_field: str, value: Any
):
    """Garantit le rejet immédiat en HTTP 422 si une colonne cible ou de fuite est injectée."""
    leaked_game = dict(valid_single_game)
    leaked_game[forbidden_field] = value

    # Enveloppé
    res_wrapped = api_client.post("/predict", json={"game": leaked_game})
    assert res_wrapped.status_code == 422
    assert res_wrapped.json()["error_code"] == "INVALID_INPUT"

    # Direct
    res_direct = api_client.post("/predict", json=leaked_game)
    assert res_direct.status_code == 422
    assert res_direct.json()["error_code"] == "INVALID_INPUT"


def test_rejection_target_leakage_in_batch(
    api_client: TestClient, sample_batch_games: List[Dict[str, Any]]
):
    """Vérifie le rejet HTTP 422 de tout le lot si un seul élément contient une variable interdite."""
    corrupted_batch = [dict(g) for g in sample_batch_games]
    corrupted_batch[1]["pct_pos_total"] = 92.0

    res = api_client.post("/predict/batch", json={"games": corrupted_batch})
    assert res.status_code == 422
    assert res.json()["error_code"] == "INVALID_INPUT"


# ==============================================================================
# 6. Bornes et Validation des Lots
# ==============================================================================

def test_rejection_empty_batch(api_client: TestClient):
    """Rejette les lots vides avec HTTP 422."""
    res = api_client.post("/predict/batch", json={"games": []})
    assert res.status_code == 422
    assert res.json()["error_code"] == "INVALID_INPUT"


def test_rejection_oversized_batch(api_client: TestClient, valid_single_game: Dict[str, Any]):
    """Rejette les lots excédant 100 éléments avec HTTP 422."""
    oversized = [dict(valid_single_game, appid=1000 + i) for i in range(101)]
    res = api_client.post("/predict/batch", json={"games": oversized})
    assert res.status_code == 422
    assert res.json()["error_code"] == "INVALID_INPUT"


# ==============================================================================
# 7. Corrélation et Propagation des Identifiants
# ==============================================================================

def test_custom_request_id_propagation(api_client: TestClient, valid_single_game: Dict[str, Any]):
    """Vérifie la prise en compte et la propagation d'un X-Request-ID client."""
    custom_req_id = "req-client-audit-987654321"
    res = api_client.post(
        "/predict",
        json={"game": valid_single_game},
        headers={"X-Request-ID": custom_req_id},
    )
    assert res.status_code == 200
    assert res.headers["X-Request-ID"] == custom_req_id
    assert res.json()["request_id"] == custom_req_id


# ==============================================================================
# 8. Comportement en Mode Non-Prêt (503)
# ==============================================================================

def test_service_unready_behavior(api_client: TestClient, valid_single_game: Dict[str, Any]):
    """Simule une perte de disponibilité des modèles et valide le retour HTTP 503."""
    state = app.state.service_state
    original_ready = state.is_ready
    try:
        # Simulation panne / non prêt
        state.is_ready = False
        state.ready_error = "Modèle déchargé pour maintenance programmée"

        # Sonde ready -> 503
        r_ready = api_client.get("/health/ready")
        assert r_ready.status_code == 503
        assert r_ready.json()["status"] == "not_ready"

        # Inférence unitaire -> 503
        r_pred = api_client.post("/predict", json={"game": valid_single_game})
        assert r_pred.status_code == 503
        assert r_pred.json()["error_code"] == "DEPENDENCY_UNAVAILABLE"

    finally:
        # Restauration de l'état nominal
        state.is_ready = original_ready
        state.ready_error = None
