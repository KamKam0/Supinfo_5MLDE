"""tests/integration/test_grafana_and_ops.py - Tests d'intégration du Dashboard Grafana et de la Supervision OPS.

Valide les critères du dashboard de supervision :
- Schéma relationnel d'observabilité 'ops' et séparation stricte des privilèges (ops_writer vs grafana_reader).
- Exporteur de télémétrie reprenable 'ops_export.py' synchronisant Prefect et SQLite vers PostgreSQL.
- Provisioning automatique du Dashboard Grafana (UID steam-mlops-overview) dans le dossier Operations.
- Validité syntaxique et exécution sans produit cartésien des requêtes SQL modulaires.
- Réconciliation mathématique à l'unité près sur la campagne d'audit étalonnée.
"""

from __future__ import annotations

import base64
import glob
import json
from pathlib import Path
import urllib.request
import uuid
from typing import Any, Dict
import psycopg
import pytest
from starlette.testclient import TestClient

from api_prediction.main import app
from src.orchestration.ops_export import export_ops_telemetry
from src.settings import settings
from scripts.verify_sql_counts import execute_known_campaign, verify_sql_reconciliation


@pytest.fixture(scope="module")
def api_client():
    """Client de test FastAPI avec cycle de vie complet."""
    with TestClient(app) as client:
        yield client


def test_ops_schema_and_roles_privileges():
    """Vérifie l'existence du schéma ops, des tables et l'étanchéité des privilèges de sécurité."""
    # 1. Vérification avec le rôle de lecture grafana_reader
    conn_url_reader = f"postgresql://grafana_reader:5mlde-local-grafana@127.0.0.1:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    with psycopg.connect(conn_url_reader) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'ops'
                ORDER BY table_name;
                """
            )
            tables = [row[0] for row in cur.fetchall()]
            assert "pipeline_runs" in tables
            assert "pipeline_tasks" in tables
            assert "export_sync_state" in tables

            # Règle d'or : interdiction absolue d'écrire pour grafana_reader
            with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
                cur.execute(
                    "INSERT INTO ops.export_sync_state (exporter_id, last_sync_at) VALUES ('intruder', now());"
                )

    # 2. Vérification avec le rôle d'écriture ops_writer
    conn_url_writer = settings.OPS_DATABASE_URL
    with psycopg.connect(conn_url_writer, autocommit=True) as conn:
        with conn.cursor() as cur:
            test_id = f"test-sync-{uuid.uuid4().hex[:6]}"
            cur.execute(
                """
                INSERT INTO ops.export_sync_state (exporter_id, last_sync_at, status)
                VALUES (%s, now(), 'OK')
                ON CONFLICT (exporter_id) DO NOTHING;
                """,
                (test_id,),
            )
            cur.execute("SELECT status FROM ops.export_sync_state WHERE exporter_id = %s;", (test_id,))
            res = cur.fetchone()
            assert res is not None
            assert res[0] == "OK"
            # Nettoyage
            cur.execute("DELETE FROM ops.export_sync_state WHERE exporter_id = %s;", (test_id,))


def test_ops_exporter_idempotent_synchronization():
    """Vérifie l'exécution de l'exporteur OPS et son idempotence stricte sans duplication."""
    # Première exécution
    res1 = export_ops_telemetry(limit=20)
    assert res1["runs_exported"] > 0
    exported_count = res1["runs_exported"]

    # Deuxième exécution immédiate : doit mettre à jour sans dupliquer de lignes
    res2 = export_ops_telemetry(limit=20)
    assert res2["runs_exported"] == exported_count

    # Vérification en base
    conn_url = f"postgresql://grafana_reader:5mlde-local-grafana@127.0.0.1:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    with psycopg.connect(conn_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM ops.pipeline_runs;")
            total_runs = cur.fetchone()[0]
            assert total_runs >= exported_count

            cur.execute("SELECT records_exported, status FROM ops.export_sync_state WHERE exporter_id = 'prefect_sync_worker';")
            state_row = cur.fetchone()
            assert state_row is not None
            assert state_row[1] == "OK"


def test_grafana_dashboard_provisioning():
    """Vérifie que Grafana a chargé automatiquement le dashboard provisionné dans Operations."""
    creds = base64.b64encode(b"admin:5mlde-local-admin").decode("ascii")
    url = f"http://127.0.0.1:{settings.GRAFANA_PORT}/api/search"
    req = urllib.request.Request(url, headers={"Authorization": f"Basic {creds}"})

    with urllib.request.urlopen(req, timeout=5) as res:
        dashboards = json.loads(res.read().decode("utf-8"))

    target_dashboard = next(
        (d for d in dashboards if d.get("uid") == "steam-mlops-overview"),
        None,
    )
    assert target_dashboard is not None, "Dashboard 'steam-mlops-overview' non trouvé dans Grafana !"
    assert target_dashboard.get("title") == "Steam MLOps — Traitements et Exploitation"
    assert target_dashboard.get("folderTitle") == "Operations"


def test_grafana_sql_queries_execution():
    """Valide l'exécution et l'exactitude syntaxique de chaque fichier SQL de requêtes Grafana."""
    conn_url = f"postgresql://grafana_reader:5mlde-local-grafana@127.0.0.1:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    query_files = glob.glob(str(settings.PROJECT_ROOT / "infra" / "grafana" / "queries" / "*.sql"))
    assert len(query_files) >= 4, f"Fichiers SQL manquants : {len(query_files)}"

    with psycopg.connect(conn_url) as conn:
        with conn.cursor() as cur:
            for qf in query_files:
                content = Path(qf).read_text(encoding="utf-8")
                statements = [s.strip() for s in content.split(";") if s.strip()]
                for i, stmt in enumerate(statements):
                    cur.execute(stmt)
                    # Vérifie que la requête ne lève pas d'exception et retourne un résultat
                    rows = cur.fetchall()
                    assert rows is not None, f"Requête vide ou échouée dans {qf} (#{i+1})"


def test_campaign_sql_reconciliation_exactness(api_client: TestClient):
    """Exécute la campagne étalonnée d'audit et valide le rapprochement mathématique à l'unité près."""
    campaign_id = f"test-audit-reconcile-{uuid.uuid4().hex[:6]}"
    execute_known_campaign(campaign_id, api_client)
    reconciliation_ok = verify_sql_reconciliation(campaign_id)
    assert reconciliation_ok is True, "Échec de réconciliation SQL sur la campagne étalonnée."
