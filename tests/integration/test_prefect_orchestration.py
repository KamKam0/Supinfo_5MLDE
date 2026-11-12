"""tests/integration/test_prefect_orchestration.py - Tests d'intégration de l'orchestration Prefect 3.

Valide :
1. L'exécution nominale du flow steam-training-pipeline avec ses 4 tâches mères visibles.
2. La corrélation universelle des identifiants (batch_id -> flow_run_id -> mlflow_run_id -> publication_id).
3. L'interruption immédiate en cas d'échec Great Expectations (retries=0, aucun run orphelin).
4. La finalisation déterministe des runs MLflow via les hooks de cycle de vie (on_flow_completion, on_flow_failure).
5. L'interfaçage réussi du worker PUSH déclenchant l'entraînement Prefect complet.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Dict
from uuid import UUID
import mlflow
import pandas as pd
from prefect import flow, task
from prefect.client.orchestration import get_client
from prefect.client.schemas.filters import FlowRunFilter, FlowRunFilterId
import pytest

from api_trigger_passif.storage import PushStorage
from api_trigger_passif.training_trigger import process_batch
from simulateur_push import generate_synthetic_push_event
from src.contracts import RunContext
from src.exceptions import DataSchemaError, DataValidationError, SteamMLOpsError
from src.orchestration.lifecycle import (
    build_correlation_record,
    format_correlation_chain,
    on_flow_completion,
    on_flow_failure,
)
from src.orchestration.tasks import (
    steam_training_pipeline,
    task_evaluate_and_publish,
    task_load_data,
    task_preprocess_data,
    task_train_model,
)
from src.settings import settings


@pytest.fixture(scope="module", autouse=True)
def setup_prefect_env():
    """Configure les variables d'environnement Prefect et MLflow pour les tests."""
    old_prefect = os.environ.get("PREFECT_API_URL")
    old_mlflow = os.environ.get("MLFLOW_TRACKING_URI")
    os.environ["PREFECT_API_URL"] = settings.PREFECT_API_URL
    os.environ["MLFLOW_TRACKING_URI"] = settings.MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    yield
    if old_prefect:
        os.environ["PREFECT_API_URL"] = old_prefect
    if old_mlflow:
        os.environ["MLFLOW_TRACKING_URI"] = old_mlflow


def test_prefect_pipeline_nominal_execution_and_correlation(tmp_path: Path):
    """Vérifie l'enchaînement des 4 tâches mères et la corrélation mutuelle des identifiants."""
    batch_id = "batch-test-l10-nominal"
    project_root = Path(__file__).resolve().parent.parent.parent
    snapshot_path = project_root / "data" / "snapshots" / "historical_projection.csv"

    config = {
        "snapshot_path": str(snapshot_path),
        "validation_suite": "snapshot_integrity_v1",
        "mode": "test-orchestration",
        "batch_id": batch_id,
    }

    # Exécution du flow complet avec paramètres d'arbres allégés pour la rapidité
    result = steam_training_pipeline(
        config=config,
        register_models=True,
        qualify=True,
        promote_champion=True,
        n_estimators=5,
        max_depth=6,
    )

    # 1. Vérifications structurelles du résultat retourné
    assert result is not None
    assert "flow_run_id" in result
    flow_run_id = result["flow_run_id"]
    mlflow_run_id = result["mlflow_run_id"]
    publication_id = result.get("publication_id")

    assert result["batch_id"] == batch_id
    assert mlflow_run_id is not None
    assert publication_id is not None
    assert "steam-positive-ratio" in result["model_versions"]
    assert "steam-owner-range" in result["model_versions"]

    # 2. Vérification de la chaîne de corrélation
    correlation = result["correlation"]
    assert correlation["batch_id"] == batch_id
    assert correlation["flow_run_id"] == flow_run_id
    assert correlation["mlflow_run_id"] == mlflow_run_id
    assert correlation["publication_id"] == publication_id
    assert "batch-test-l10-nominal" in correlation["formatted_chain"]
    assert flow_run_id in correlation["formatted_chain"]

    # 3. Vérification des tags MLflow (référencement réciproque de Prefect)
    mlflow_client = mlflow.tracking.MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)
    run = mlflow_client.get_run(mlflow_run_id)
    assert run.info.status == "FINISHED"
    assert run.data.tags.get("prefect.flow_run_id") == flow_run_id
    assert run.data.tags.get("prefect.flow_name") == "steam-training-pipeline"
    assert run.data.tags.get("batch_id") == batch_id

    # 4. Vérification dans l'API Prefect Server
    async def verify_prefect_server():
        async with get_client() as client:
            fr = await client.read_flow_run(UUID(flow_run_id))
            assert fr.state.name == "Completed"

            f_filter = FlowRunFilter(id=FlowRunFilterId(any_=[UUID(flow_run_id)]))
            task_runs = await client.read_task_runs(flow_run_filter=f_filter)
            assert len(task_runs) == 4
            task_names = [t.name.split("-")[0] for t in task_runs]
            assert "task_load_data" in task_names
            assert "task_preprocess_data" in task_names
            assert "task_train_model" in task_names
            assert "task_evaluate_and_publish" in task_names

    asyncio.run(verify_prefect_server())


def test_prefect_pipeline_halts_immediately_on_gx_failure(tmp_path: Path):
    """Vérifie le blocage immédiat sans retry sur non-conformité Great Expectations."""
    corrupt_csv = tmp_path / "corrupted_snapshot.csv"
    # Création d'un dataset invalide (prix négatifs et colonnes manquantes)
    corrupted_data = pd.DataFrame([
        {"appid": 999901, "name": "Bad Game 1", "price": -50.0},
        {"appid": 999902, "name": "Bad Game 2", "price": -10.0},
    ])
    corrupted_data.to_csv(corrupt_csv, index=False)

    config = {
        "snapshot_path": str(corrupt_csv),
        "validation_suite": "snapshot_integrity_v1",
        "batch_id": "batch-corrupt",
    }

    # Le flow doit lever DataValidationError et s'interrompre sans retry
    with pytest.raises(Exception) as exc_info:
        steam_training_pipeline(config=config)

    # Vérification que l'erreur provient de la barrière de données
    assert isinstance(exc_info.value, (DataValidationError, SteamMLOpsError))


def test_prefect_lifecycle_hook_closes_mlflow_on_failure():
    """Vérifie que le hook on_flow_failure clôture proprement tout run MLflow sans laisser d'orphelin."""
    mlflow_client = mlflow.tracking.MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)
    exp = mlflow_client.get_experiment_by_name("steam-mlops-experiments")
    exp_id = exp.experiment_id if exp else "0"

    orphaned_run_id = None

    @flow(name="test-simulated-failure-flow", on_failure=[on_flow_failure])
    def failing_pipeline():
        nonlocal orphaned_run_id
        # Démarrage d'un run MLflow
        run = mlflow.start_run(experiment_id=exp_id, run_name="simulated-crash-run")
        orphaned_run_id = run.info.run_id
        # Simulation d'une défaillance inattendue en cours de traitement
        raise RuntimeError("Crash inopiné du processus d'entraînement")

    with pytest.raises(Exception):
        failing_pipeline()

    # Vérifications critiques de l'observabilité :
    # 1. Aucun run actif ne doit subsister dans le thread
    assert mlflow.active_run() is None

    # 2. Le run MLflow orphelin doit avoir été basculé en FAILED par le hook Prefect
    assert orphaned_run_id is not None
    mlflow_run = mlflow_client.get_run(orphaned_run_id)
    assert mlflow_run.info.status == "FAILED"


def test_push_worker_triggers_prefect_pipeline(tmp_path: Path):
    """Vérifie l'intégration du worker PUSH avec le flow Prefect."""
    db_path = tmp_path / "test_push_storage.db"
    storage = PushStorage(db_path=db_path)

    # Création d'un ensemble de 500 jeux synthétiques valides
    project_root = Path(__file__).resolve().parent.parent.parent
    base_snapshot = project_root / "data" / "snapshots" / "historical_projection.csv"

    games_list = [generate_synthetic_push_event(i, base_appid=3000000)["game"] for i in range(1, 501)]

    batch_id = "batch-test-prefect-worker"
    now_iso = "2026-11-16T12:00:00Z"

    # Insertion directe d'un lot PENDING de 500 jeux dans SQLite
    conn = storage._get_connection()
    with conn:
        conn.execute(
            "INSERT INTO batches (batch_id, status, created_at, row_count) VALUES (?, ?, ?, ?)",
            (batch_id, "PENDING", now_iso, 500),
        )
        for row in games_list:
            conn.execute(
                """
                INSERT OR REPLACE INTO games (
                    appid, name, release_date, required_age, price, dlc_count,
                    windows, mac, linux, achievements, genres, positive, negative,
                    pct_pos_total, num_reviews_total, estimated_owners, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["appid"], row["name"], row["release_date"], row["required_age"],
                    row["price"], row["dlc_count"], int(row["windows"]), int(row["mac"]),
                    int(row["linux"]), row["achievements"], row["genres"], row["positive"],
                    row["negative"], row["pct_pos_total"], row["num_reviews_total"],
                    row["estimated_owners"], now_iso,
                ),
            )
            conn.execute(
                "INSERT INTO batch_items (batch_id, appid, assigned_at) VALUES (?, ?, ?)",
                (batch_id, row["appid"], now_iso),
            )

    # Traitement du lot avec déclenchement de l'entraînement Prefect
    result = process_batch(
        batch_id=batch_id,
        storage=storage,
        base_snapshot_path=base_snapshot,
        snapshots_dir=tmp_path / "snapshots",
        trigger_training=True,
        training_kwargs={"n_estimators": 5, "max_depth": 6},
    )

    assert result["status"] == "SUCCEEDED"
    assert "pipeline_result" in result
    pipeline_res = result["pipeline_result"]
    assert pipeline_res is not None
    assert pipeline_res["batch_id"] == batch_id
    assert "flow_run_id" in pipeline_res
    assert "mlflow_run_id" in pipeline_res
    assert "publication_id" in pipeline_res
    assert "correlation" in pipeline_res
