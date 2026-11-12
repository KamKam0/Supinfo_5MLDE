"""tests/integration/test_caching_and_alerts.py - Tests du cache pur, des alertes et du déploiement.

Valide :
1. La politique de mise en cache déterministe (hit sur identité, invalidation sur contenu, paramètre ou code).
2. L'absence totale de fuite d'état ou d'identifiant dans les artefacts mis en cache.
3. Le fonctionnement du récepteur local autonome d'alertes HTTP (port 9000).
4. Le déclenchement automatique d'alertes sur échec critique du pipeline sans blocage résiduel.
5. Le respect absolu de la règle de sécurité Teams (zéro envoi non autorisé).
6. Le cycle de vie du déploiement Prefect (création, planification 10 min, pause et reprise).
"""

from __future__ import annotations

import asyncio
from http.server import HTTPServer
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict
from uuid import UUID
import mlflow
import pandas as pd
from prefect import flow
from prefect.client.orchestration import get_client
import pytest

from scripts.notification_receiver import AlertStore, NotificationHTTPRequestHandler, run_receiver_server
from src.orchestration.alerts import AlertDispatcher, AlertMessage, format_teams_card_payload
from src.orchestration.cache_policy import (
    FeatureCache,
    build_features_pure,
    compute_code_hash,
    compute_feature_cache_key,
)
from src.orchestration.deployment import (
    apply_deployment,
    pause_deployment_schedule,
    resume_deployment_schedule,
)
from src.orchestration.lifecycle import on_flow_failure
from src.settings import settings


@pytest.fixture(scope="module", autouse=True)
def setup_env():
    """Configure l'environnement de test."""
    old_prefect = os.environ.get("PREFECT_API_URL")
    os.environ["PREFECT_API_URL"] = settings.PREFECT_API_URL
    os.environ["MLFLOW_TRACKING_URI"] = settings.MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    yield
    if old_prefect:
        os.environ["PREFECT_API_URL"] = old_prefect


@pytest.fixture
def local_receiver(tmp_path: Path):
    """Démarre un serveur de notification local éphémère sur un port dédié."""
    storage_file = tmp_path / "test_alerts.json"
    server_port = 9005  # Port de test isolé
    server = run_receiver_server(host="127.0.0.1", port=server_port, storage_path=storage_file)

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    time.sleep(0.1)  # Laisser le socket s'initialiser

    receiver_url = f"http://127.0.0.1:{server_port}/alerts"
    yield receiver_url, storage_file

    server.shutdown()
    server.server_close()


# ==============================================================================
# 1. Tests de la Politique de Cache de Calcul Pur
# ==============================================================================


def test_feature_cache_deterministic_hit_and_invalidations(tmp_path: Path):
    """Vérifie : réutilisation sur données identiques, et invalidation immédiate sur altération."""
    cache_dir = tmp_path / "cache_features"
    cache = FeatureCache(cache_dir=cache_dir)

    df_base = pd.DataFrame([
        {
            "price": 9.99,
            "required_age": 0,
            "dlc_count": 1,
            "achievements": 10,
            "windows": True,
            "mac": False,
            "linux": False,
            "genres": "['Action', 'Indie']",
            "release_date": "2022-01-01",
        },
        {
            "price": 19.99,
            "required_age": 18,
            "dlc_count": 0,
            "achievements": 5,
            "windows": True,
            "mac": True,
            "linux": False,
            "genres": "['Strategy']",
            "release_date": "2023-05-10",
        },
    ])

    # 1. Premier passage : Cache MISS et persistance
    res1, hit1, key1 = build_features_pure(df_base, top_genres_k=5, cache_dir=cache_dir)
    assert hit1 is False
    assert (cache_dir / f"{key1}.parquet").exists()

    # 2. Deuxième passage identique : Cache HIT immédiat
    res2, hit2, key2 = build_features_pure(df_base, top_genres_k=5, cache_dir=cache_dir)
    assert hit2 is True
    assert key2 == key1
    pd.testing.assert_frame_equal(res1, res2)

    # 3. Invalidation 1 : Modification d'un paramètre (top_genres_k=3 au lieu de 5)
    res3, hit3, key3 = build_features_pure(df_base, top_genres_k=3, cache_dir=cache_dir)
    assert hit3 is False
    assert key3 != key1

    # 4. Invalidation 2 : Modification du contenu des données sous le même schéma
    df_altered = df_base.copy()
    df_altered.loc[0, "price"] = 49.99
    res4, hit4, key4 = build_features_pure(df_altered, top_genres_k=5, cache_dir=cache_dir)
    assert hit4 is False
    assert key4 != key1

    # 5. Invalidation 3 : Modification du code source (hash de code différent)
    res5, hit5, key5 = build_features_pure(
        df_base,
        top_genres_k=5,
        cache_dir=cache_dir,
        custom_code_hash="mock_source_code_hash_v2",
    )
    assert hit5 is False
    assert key5 != key1

    # 6. Absence totale de run_id, fuite ou cible dans le cache
    assert "pct_pos_total" not in res1.columns
    assert "estimated_owners" not in res1.columns
    assert "run_id" not in res1.columns
    assert "flow_run_id" not in res1.columns


# ==============================================================================
# 2. Tests du Récepteur d'Alertes et du Dispatcheur
# ==============================================================================


def test_alert_dispatcher_delivers_to_local_receiver(local_receiver):
    """Vérifie la transmission et l'archivage fidèle d'une alerte sur le récepteur local."""
    receiver_url, storage_file = local_receiver

    alert = AlertMessage(
        flow_name="steam-training-pipeline",
        flow_run_id="test-flow-uuid-1234",
        state="FAILED",
        error_message="Panne simulée : Great Expectations assertion failed",
        environment="test",
        mode="scheduled-run",
        batch_id="batch-alert-001",
        mlflow_run_id="mlflow-run-5678",
    )

    dispatcher = AlertDispatcher(local_webhook_url=receiver_url, teams_enabled=False)
    dispatch_res = dispatcher.send_alert(alert)

    assert dispatch_res["local"]["sent"] is True
    assert dispatch_res["local"]["status"] == "DELIVERED"
    assert dispatch_res["teams"]["status"] == "SKIPPED"

    # Vérification dans le fichier de persistance du récepteur
    store = AlertStore(storage_path=storage_file)
    alerts = store.get_alerts()
    assert len(alerts) == 1
    stored_payload = alerts[0]["payload"]
    assert stored_payload["flow_name"] == "steam-training-pipeline"
    assert stored_payload["flow_run_id"] == "test-flow-uuid-1234"
    assert stored_payload["state"] == "FAILED"
    assert stored_payload["batch_id"] == "batch-alert-001"
    assert stored_payload["mlflow_run_id"] == "mlflow-run-5678"


def test_pipeline_failure_triggers_alert_via_lifecycle_hook(local_receiver):
    """Vérifie que on_flow_failure transmet automatiquement l'alerte au récepteur local."""
    receiver_url, storage_file = local_receiver
    os.environ["LOCAL_ALERT_WEBHOOK_URL"] = receiver_url

    failed_flow_run_id = None

    @flow(name="test-monitored-pipeline", on_failure=[on_flow_failure])
    def failing_flow():
        nonlocal failed_flow_run_id
        from prefect.runtime import flow_run
        failed_flow_run_id = str(flow_run.id)
        raise RuntimeError("Interruption critique détectée dans le worker")

    with pytest.raises(Exception):
        failing_flow()

    # Vérification que le récepteur a bien enregistré l'alerte
    store = AlertStore(storage_path=storage_file)
    alerts = store.get_alerts()
    assert len(alerts) >= 1
    last_alert = alerts[-1]["payload"]
    assert last_alert["flow_name"] == "test-monitored-pipeline"
    assert last_alert["flow_run_id"] == failed_flow_run_id
    assert "Interruption critique" in last_alert["error_message"]


def test_alert_dispatcher_resilience_when_receiver_offline():
    """Vérifie qu'un récepteur inaccessible ne provoque aucune exception bloquante."""
    offline_url = "http://127.0.0.1:9999/unreachable_endpoint"
    dispatcher = AlertDispatcher(local_webhook_url=offline_url, teams_enabled=False, timeout=1.0)

    alert = AlertMessage(
        flow_name="steam-training-pipeline",
        flow_run_id="test-resilience",
        error_message="Test non-blocking error",
    )

    # L'appel ne doit PAS lever d'exception
    res = dispatcher.send_alert(alert)
    assert res["local"]["sent"] is False
    assert res["local"]["status"] in ("CONNECTION_FAILED", "REJECTED")


def test_teams_card_formatting_and_security_gate():
    """Vérifie le formatage Teams et l'interdiction stricte de requête non autorisée."""
    alert = AlertMessage(
        flow_name="steam-training-pipeline",
        flow_run_id="test-teams-uuid",
        error_message="Test Teams Formatting",
        batch_id="batch-123",
        mlflow_run_id="mlflow-456",
    )

    # 1. Vérification du formatage Adaptive Card / MessageCard
    card = format_teams_card_payload(alert)
    assert card["@type"] == "MessageCard"
    assert card["themeColor"] == "D63031"
    facts = card["sections"][0]["facts"]
    fact_dict = {f["name"]: f["value"] for f in facts}
    assert fact_dict["État terminal :"] == "FAILED"
    assert fact_dict["Lot de données (batch_id) :"] == "batch-123"
    assert fact_dict["Run MLflow associé :"] == "mlflow-456"

    # 2. Vérification de la porte de sécurité : TEAMS_ALERT_ENABLED=false
    dispatcher = AlertDispatcher(
        teams_webhook_url="https://outlook.office.com/webhook/dummy",
        teams_enabled=False,
    )
    res = dispatcher.send_alert(alert)
    assert res["teams"]["sent"] is False
    assert res["teams"]["status"] == "SKIPPED"
    assert "Teams désactivé par défaut" in res["teams"]["reason"]


# ==============================================================================
# 3. Tests du Déploiement et de la Planification Prefect
# ==============================================================================


def test_prefect_deployment_and_schedule_management():
    """Vérifie l'enregistrement du déploiement 'steam-batch-deployment', pause et reprise."""
    dep_name = "steam-batch-deployment"

    # 1. Application du déploiement avec intervalle de 10 minutes
    dep_id_str = apply_deployment(name=dep_name, interval_minutes=10)
    assert dep_id_str is not None

    async def verify_deployment_in_server():
        async with get_client() as client:
            dep = await client.read_deployment(UUID(dep_id_str))
            assert dep.name == dep_name
            assert dep.paused is False
            assert len(dep.schedules) == 1

            # 2. Mise en pause du planning
            paused_ok = await pause_deployment_schedule(dep_name)
            assert paused_ok is True
            dep_paused = await client.read_deployment(UUID(dep_id_str))
            assert dep_paused.paused is True

            # 3. Reprise du planning
            resumed_ok = await resume_deployment_schedule(dep_name)
            assert resumed_ok is True
            dep_resumed = await client.read_deployment(UUID(dep_id_str))
            assert dep_resumed.paused is False

            # 4. Rétablissement en pause pour clore proprement la démonstration
            await pause_deployment_schedule(dep_name)

    asyncio.run(verify_deployment_in_server())
