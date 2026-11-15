"""tests/integration/test_incidents_and_recovery.py - Tests de résilience, reprise et rollback.

Valide :
- L'interception bloquante des lots corrompus par Great Expectations Core 1.x.
- La bascule en HTTP 503 et le repli en télémétrie locale lors d'une indisponibilité PostgreSQL.
- L'idempotence stricte et l'absence de versions dupliquées lors de la réconciliation MLflow.
- Le cycle de vie complet du retour arrière (Rollback) sans réentraînement.
- Le confinement des erreurs du dispatcheur d'alertes en cas de webhook injoignable.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Dict
from unittest.mock import MagicMock
from uuid import UUID, uuid4
import pytest

from fastapi.testclient import TestClient
import pandas as pd

from api_prediction.lifecycle import verify_manifest_hash
from api_prediction.main import app
from api_prediction.repository import DatabasePool, db_pool
from src.data_validation import validate_dataframe
from src.exceptions import DataValidationError
from src.orchestration.alerts import AlertDispatcher, AlertMessage, trigger_failure_alert
from src.settings import settings
from src.tracking.publications import PublicationManager
from src.tracking.reconciliation import reconcile_publication
from src.tracking.registry import get_client, rollback_model_alias


def test_incident_invalid_data_triggers_gx_barrier_and_preserves_models():
    """Vérifie que des données invalides déclenchent DataValidationError sans corrompre les modèles existants."""
    manifest_path = settings.PROJECT_ROOT / "state" / "service-manifest.json"
    manifest_sha_before = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    # Données corrompues : prix textuel et pourcentage hors bornes
    df_corrupted = pd.DataFrame([
        {
            "appid": 999888,
            "name": "Invalid Game",
            "release_date": "2023-01-01",
            "price": "Gratuit",  # Viol de type
            "positive": 10,
            "negative": 0,
            "pct_pos_total": 250.0,  # Viol de borne (> 100)
            "estimated_owners": "1000 - 2000",
            "num_reviews_total": 10,
            "required_age": 0,
            "dlc_count": 0,
            "achievements": 0,
            "windows": True,
            "mac": False,
            "linux": False,
            "genres": "Indie",
        }
    ])

    with pytest.raises(DataValidationError):
        validate_dataframe(
            df_corrupted,
            suite_name="historical_schema_v1",
            dataset_name="test_corrupted_unit",
            raise_on_error=True,
        )

    # Vérification que le manifeste de service actif n'a pas bougé d'un bit
    manifest_sha_after = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert manifest_sha_before == manifest_sha_after


def test_incident_postgres_unavailability_returns_503_and_logs_fallback(monkeypatch):
    """Vérifie le mode dégradé strict de l'API en cas de coupure PostgreSQL (aucun faux 200)."""
    test_game = {
        "appid": 1086940,
        "name": "Baldur's Gate 3",
        "price": 59.99,
        "required_age": 18,
        "genres": ["RPG", "Strategy", "Adventure"],
        "windows": True,
        "mac": True,
        "linux": False,
        "release_date": "2023-08-03",
    }

    fallback_file = settings.PROJECT_ROOT / "state" / "telemetry_fallback.jsonl"
    lines_before = len(fallback_file.read_text(encoding="utf-8").splitlines()) if fallback_file.exists() else 0

    with TestClient(app) as client:
        # Sonde live conforme
        resp_live = client.get("/health/live")
        assert resp_live.status_code == 200

        # Simulation de la rupture PostgreSQL
        monkeypatch.setattr(db_pool, "is_healthy", lambda: False)

        def mock_broken_connection(*args, **kwargs):
            raise ConnectionError("PostgreSQL indisponible (simulation de coupure)")

        monkeypatch.setattr(db_pool, "connection", mock_broken_connection)

        # 1. Vérification que la sonde d'aptitude /health/ready bascule en 503
        resp_ready = client.get("/health/ready")
        assert resp_ready.status_code == 503
        assert resp_ready.json()["status"] == "not_ready"

        # 2. Vérification que l'inférence /predict est rejetée en 503 (zéro faux succès 200)
        resp_pred = client.post("/predict", json={"game": test_game})
        assert resp_pred.status_code == 503
        data_pred = resp_pred.json()
        assert "detail" in data_pred or data_pred.get("error_code") == "DEPENDENCY_UNAVAILABLE"

        # 3. Vérification de la télémétrie locale de secours
        assert fallback_file.exists()
        lines_after = len(fallback_file.read_text(encoding="utf-8").splitlines())
        assert lines_after > lines_before


def test_incident_mlflow_reconciliation_is_idempotent():
    """Valide la reprise d'une publication partielle et l'absence totale de duplication."""
    pub_manager = PublicationManager()
    manifest_path = settings.PROJECT_ROOT / "state" / "service-manifest.json"
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_id = manifest_data["models"]["regression"]["mlflow_run_id"]
    reg_ver = manifest_data["models"]["regression"]["version"]

    pub_id = f"pub-unit-reconcile-{uuid4().hex[:6]}"
    target_alias = f"test-unit-alias-{uuid4().hex[:4]}"

    # Déclaration d'une publication interrompue (régression ALIASED, classification en attente)
    pub_manager.create_publication(
        run_id=run_id,
        verdict="ACCEPTED",
        target_alias=target_alias,
        publication_id=pub_id,
        regression_model_name="steam-positive-ratio",
        classification_model_name="steam-owner-range",
        regression_model_uri=f"runs:/{run_id}/model_regression",
        classification_model_uri=f"runs:/{run_id}/model_classification",
    )
    pub_manager.update_publication(
        publication_id=pub_id,
        status="IN_PROGRESS",
        regression_version=reg_ver,
        regression_status="ALIASED",
        classification_version=None,
        classification_status=None,
    )

    # 1. Première réconciliation : doit compléter la classification
    res_1 = reconcile_publication(pub_id, pub_manager=pub_manager)
    assert res_1["status"] == "COMPLETED"
    assert res_1["already_completed"] is False
    assert res_1["classification_version"] is not None

    pub_db = pub_manager.get_publication(pub_id)
    assert pub_db["status"] == "COMPLETED"

    # 2. Deuxième réconciliation : doit immédiatement retourner already_completed sans rien toucher
    res_2 = reconcile_publication(pub_id, pub_manager=pub_manager)
    assert res_2["already_completed"] is True
    assert res_2["status"] == "COMPLETED"
    assert res_2["reconciled_steps"] == []


def test_incident_model_rollback_and_hash_tampering_protection():
    """Valide la détection d'altération de hash et la restauration de service-manifest.prev.json."""
    manifest_path = settings.PROJECT_ROOT / "state" / "service-manifest.json"
    prev_manifest_path = settings.PROJECT_ROOT / "state" / "service-manifest.prev.json"
    assert manifest_path.exists()

    # Sauvegarde
    if not prev_manifest_path.exists():
        shutil.copy2(manifest_path, prev_manifest_path)

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))

    # 1. Vérification du hash nominal
    assert verify_manifest_hash(manifest_data) is True

    # 2. Altération délibérée : modification du contenu sans recalculer le hash
    corrupted_manifest = dict(manifest_data)
    corrupted_manifest["mode"] = "tampered_mode"
    assert verify_manifest_hash(corrupted_manifest) is False

    # 3. Test de Rollback : restauration du manifeste sain
    shutil.copy2(prev_manifest_path, manifest_path)
    restored_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert verify_manifest_hash(restored_data) is True
    assert restored_data["manifest_id"] == manifest_data["manifest_id"]


def test_incident_alert_dispatcher_handles_connection_failure():
    """Vérifie que l'échec de joignabilité du récepteur d'alertes est confiné sans crash."""
    dispatcher = AlertDispatcher(
        local_webhook_url="http://127.0.0.1:59998/closed_port_test",
        teams_enabled=False,
        timeout=0.5,
    )

    alert = AlertMessage(
        flow_name="test-flow",
        flow_run_id=f"flow-test-{uuid4().hex[:6]}",
        timestamp="2026-11-16T12:00:00Z",
        environment="test",
        mode="test",
        state="FAILED",
        error_message="Test containment",
    )

    # Ne doit pas lever d'exception
    res = dispatcher.send_alert(alert)
    assert res["local"]["sent"] is False
    assert res["local"]["status"] == "CONNECTION_FAILED"
    assert "error" in res["local"]
