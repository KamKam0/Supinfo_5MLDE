"""tests/integration/test_load_benchmarks.py - Tests unitaires et d'intégration du banc de charge.

Valide :
- L'intégrité et l'étanchéité anti-fuite des fixtures de charge 'fixtures/load_fixtures.json'.
- Les trois profils d'utilisateurs virtuels Locust (Single, Batch, Mixed).
- La détection stricte des faux 200 par le validateur de réponse.
- Le cycle de vie du collecteur de ressources système 'ResourceMonitor'.
- L'analyseur de capacité et le générateur du rapport de dimensionnement 'reports/sizing.md'.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from scripts.analyse_capacity import analyze_single_campaign, generate_sizing_report
from scripts.collect_resources import ResourceMonitor
from src.settings import settings
from tests.load.locust_helpers import (
    BATCH_20_GAMES,
    LOAD_GAMES,
    get_campaign_header,
    validate_single_prediction_response,
)


def test_load_fixtures_integrity():
    """Vérifie l'existence, l'empreinte et l'absence totale de variables cibles dans les fixtures."""
    fixtures_file = settings.PROJECT_ROOT / "fixtures" / "load_fixtures.json"
    assert fixtures_file.exists(), "Fichier fixtures/load_fixtures.json manquant !"

    data = json.loads(fixtures_file.read_text(encoding="utf-8"))
    games = data.get("games", [])
    assert len(games) >= 20, f"Nombre insuffisant de fixtures : {len(games)} (minimum 20)"

    # Empreinte SHA-256
    computed_sha = hashlib.sha256(fixtures_file.read_bytes()).hexdigest()
    cfg_file = settings.PROJECT_ROOT / "config" / "loadtest.json"
    if cfg_file.exists():
        cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
        expected_sha = cfg.get("fixtures_sha256")
        assert computed_sha == expected_sha, "Empreinte SHA-256 des fixtures en désaccord avec la configuration !"

    # Règle d'or absolue anti-fuite
    forbidden_keys = {"pct_pos_total", "estimated_owners", "num_reviews_total", "positive", "negative"}
    for g in games:
        found_leaks = set(g.keys()).intersection(forbidden_keys)
        assert not found_leaks, f"Fuite de variable cible détectée dans la fixture {g.get('name')} : {found_leaks}"
        assert g.get("appid") is not None and g.get("appid") > 0
        assert g.get("price") is not None and g.get("price") >= 0.0


def test_locust_user_classes_instantiation_and_tasks():
    """Valide les classes utilisateurs virtuels Locust et la génération des en-têtes."""
    import subprocess
    import sys

    # Valide que locustfile se charge sans erreur avec ses classes utilisateurs (isolé pour éviter conflit monkeypatch gevent/ssl)
    cmd = [
        sys.executable,
        "-c",
        "from tests.load.locustfile import SinglePredictionUser, BatchPredictionUser, MixedWorkloadUser; "
        "assert SinglePredictionUser is not None; assert BatchPredictionUser is not None; assert MixedWorkloadUser is not None; print('OK')",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "OK" in proc.stdout

    # Contrôle de l'en-tête X-Test-Campaign
    headers = get_campaign_header()
    assert "X-Test-Campaign" in headers
    assert len(headers["X-Test-Campaign"]) <= 128

    # Vérification du lot de 20 jeux
    assert len(BATCH_20_GAMES) == 20


def test_locust_response_validator_detects_false_200():
    """Vérifie que validate_single_prediction_response détecte et sanctionne les faux 200."""
    # 1. Cas nominal conforme
    mock_resp_ok = MagicMock()
    mock_resp_ok.status_code = 200
    mock_resp_ok.json.return_value = {
        "request_id": "req-nominal",
        "item_count": 1,
        "results": [
            {
                "item_index": 0,
                "predicted_pct": 82.5,
                "predicted_owners": "1000000 - 2000000",
                "predictions": [
                    {"task": "regression", "prediction_id": "p-reg-1"},
                    {"task": "classification", "prediction_id": "p-cls-1"},
                ],
            }
        ],
    }
    res_ids = validate_single_prediction_response(mock_resp_ok, expected_item_count=1)
    assert res_ids == ["p-reg-1", "p-cls-1"]
    mock_resp_ok.success.assert_called_once()
    mock_resp_ok.failure.assert_not_called()

    # 2. Faux 200 : champ results manquant
    mock_resp_missing_results = MagicMock()
    mock_resp_missing_results.status_code = 200
    mock_resp_missing_results.json.return_value = {"request_id": "req-broken", "item_count": 1}
    validate_single_prediction_response(mock_resp_missing_results, expected_item_count=1)
    mock_resp_missing_results.failure.assert_called_once()

    # 3. Faux 200 : pourcentage hors bornes [0, 100]
    mock_resp_bad_pct = MagicMock()
    mock_resp_bad_pct.status_code = 200
    mock_resp_bad_pct.json.return_value = {
        "request_id": "req-bad-pct",
        "item_count": 1,
        "results": [
            {
                "item_index": 0,
                "predicted_pct": 150.0,  # Invalide !
                "predicted_owners": "100 - 200",
                "predictions": [
                    {"task": "regression", "prediction_id": "p-1"},
                    {"task": "classification", "prediction_id": "p-2"},
                ],
            }
        ],
    }
    validate_single_prediction_response(mock_resp_bad_pct, expected_item_count=1)
    mock_resp_bad_pct.failure.assert_called_once()


def test_resource_monitor_lifecycle(tmp_path: Path):
    """Vérifie le démarrage, l'échantillonnage et la synthèse de ResourceMonitor."""
    csv_file = tmp_path / "test_resources.csv"
    monitor = ResourceMonitor(output_csv_path=csv_file, interval_seconds=0.2)

    monitor.start()
    import time
    time.sleep(0.5)
    summary = monitor.stop()

    assert csv_file.exists()
    assert summary["samples_count"] >= 2
    assert summary["max_system_cpu_pct"] >= 0.0
    assert summary["max_system_cpu_pct"] <= 100.0


def test_capacity_analysis_and_sizing_report_generation(tmp_path: Path):
    """Vérifie l'analyse de campagne synthétique et la génération du rapport Markdown."""
    # Création d'un dossier de tir fictif conforme
    fake_camp_dir = tmp_path / "loadtest-fake-01"
    fake_camp_dir.mkdir(parents=True, exist_ok=True)

    camp_meta = {
        "campaign_id": "loadtest-fake-01",
        "scenario": "single",
        "workers_count": 2,
        "target_users": 10,
        "actual_duration_seconds": 30.0,
        "resources_summary": {
            "avg_uvicorn_cpu_pct": 45.2,
            "avg_uvicorn_rss_mb": 195.4,
        },
    }
    with open(fake_camp_dir / "campaign.json", "w", encoding="utf-8") as f:
        json.dump(camp_meta, f)

    locust_stats_content = """Type,Name,Request Count,Failure Count,Median Response Time,Average Response Time,Min Response Time,Max Response Time,Average Content Size,Requests/s,Failures/s,50%,66%,75%,80%,90%,95%,98%,99%,99.9%,99.99%,100%
POST,/predict,300,0,25.0,28.5,12.0,110.0,850,10.0,0.0,25,30,35,40,55,75,90,100,105,110,110
None,Aggregated,300,0,25.0,28.5,12.0,110.0,850,10.0,0.0,25,30,35,40,55,75,90,100,105,110,110
"""
    (fake_camp_dir / "locust_stats.csv").write_text(locust_stats_content, encoding="utf-8")

    # Analyse
    analysis = analyze_single_campaign(fake_camp_dir)
    assert analysis is not None
    assert analysis["rps"] == 10.0
    assert analysis["p95_ms"] == 75.0
    assert analysis["error_rate_pct"] == 0.0
    assert analysis["slo_compliant"] is True

    # Génération du rapport
    output_md = tmp_path / "test_sizing.md"
    rep = generate_sizing_report(
        analyses=[analysis],
        output_path=output_md,
        safe_margin=0.30,
        peak_qps_hypothesis=100.0,
    )
    assert output_md.exists()
    assert "Rapport de Charge et Recommandation de Dimensionnement" in rep
    assert "Configuration 2 Worker(s)" in rep
