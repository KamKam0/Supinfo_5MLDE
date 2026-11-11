"""tests/unit/test_qualification.py - Tests unitaires de la politique de qualification et gouvernance.

Vérifie :
1. Le déterminisme strict de la fonction de qualification (PRODUCTION_READY, REJECTED, DEMO_ONLY).
2. Le rejet systématique lors d'échecs sur baselines (MAE ou F1-macro), NaN/Inf ou métriques manquantes.
3. Le bridage des modèles synthétiques en DEMO_ONLY (règle de provenance).
4. Le journal transactionnel des publications et la traçabilité des événements d'audit.
5. La réconciliation d'une publication interrompue sans duplication de version.
6. La procédure de retour arrière (rollback) vers une version antérieure avec audit complet.
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.tracking.publications import PublicationManager
from src.tracking.qualification import evaluate_run_qualification
from src.tracking.reconciliation import reconcile_publication
from src.tracking.registry import (
    get_model_version_by_alias,
    promote_model_to_alias,
    rollback_model_alias,
    set_model_alias,
)


@pytest.fixture
def nominal_metrics():
    """Jeu de métriques valides surpassant les baselines naïves."""
    return {
        "regression.val.mae": 12.50,
        "regression.val.baseline_mae": 13.90,
        "regression.val.rmse": 17.50,
        "regression.val.r2": 0.1650,
        "classification.val.accuracy": 0.7350,
        "classification.val.baseline_accuracy": 0.7300,
        "classification.val.f1_macro": 0.1050,
        "classification.val.baseline_f1_macro": 0.0700,
    }


# ==============================================================================
# 1. Tests Déterministes de la Politique de Qualification
# ==============================================================================

def test_qualification_nominal_production_ready(nominal_metrics):
    """Un run dépassant les baselines avec métriques finies et rechargement OK est PRODUCTION_READY."""
    report = evaluate_run_qualification(
        run_id="run-nominal-001",
        metrics=nominal_metrics,
        tags={"pipeline_mode": "HISTORICAL"},
        reload_test_passed=True,
    )
    assert report.verdict == "PRODUCTION_READY"
    assert report.passed is True
    assert len(report.rejection_reasons) == 0
    assert all(c.passed for c in report.checks)


def test_qualification_rejects_worse_regression_mae(nominal_metrics):
    """Un run dont la MAE est supérieure ou égale à la baseline est REJECTED."""
    bad_metrics = nominal_metrics.copy()
    bad_metrics["regression.val.mae"] = 14.50  # > baseline 13.90

    report = evaluate_run_qualification(
        run_id="run-bad-reg",
        metrics=bad_metrics,
        tags={"pipeline_mode": "HISTORICAL"},
        reload_test_passed=True,
    )
    assert report.verdict == "REJECTED"
    assert report.passed is False
    assert any("Régression MAE" in r for r in report.rejection_reasons)


def test_qualification_rejects_worse_classification_f1(nominal_metrics):
    """Un run dont le F1-macro est inférieur ou égal à la baseline est REJECTED."""
    bad_metrics = nominal_metrics.copy()
    bad_metrics["classification.val.f1_macro"] = 0.0650  # < baseline 0.0700

    report = evaluate_run_qualification(
        run_id="run-bad-cls",
        metrics=bad_metrics,
        tags={"pipeline_mode": "HISTORICAL"},
        reload_test_passed=True,
    )
    assert report.verdict == "REJECTED"
    assert report.passed is False
    assert any("Classification F1-Macro" in r for r in report.rejection_reasons)


def test_qualification_rejects_missing_mandatory_metrics(nominal_metrics):
    """L'absence d'une métrique obligatoire entraîne un REJECTED immédiat."""
    incomplete = nominal_metrics.copy()
    del incomplete["regression.val.r2"]

    report = evaluate_run_qualification(
        run_id="run-missing-metric",
        metrics=incomplete,
        tags={"pipeline_mode": "HISTORICAL"},
        reload_test_passed=True,
    )
    assert report.verdict == "REJECTED"
    assert report.passed is False
    assert any("Métriques obligatoires manquantes" in r for r in report.rejection_reasons)


def test_qualification_rejects_non_finite_metrics(nominal_metrics):
    """La présence d'un NaN ou d'un Inf entraîne un REJECTED strict."""
    nan_metrics = nominal_metrics.copy()
    nan_metrics["regression.val.rmse"] = float("nan")

    report = evaluate_run_qualification(
        run_id="run-nan",
        metrics=nan_metrics,
        tags={"pipeline_mode": "HISTORICAL"},
        reload_test_passed=True,
    )
    assert report.verdict == "REJECTED"
    assert report.passed is False
    assert any("non finies" in r for r in report.rejection_reasons)


def test_qualification_rejects_external_reload_failure(nominal_metrics):
    """Si le rechargement externe a échoué, le modèle est REJECTED."""
    report = evaluate_run_qualification(
        run_id="run-reload-fail",
        metrics=nominal_metrics,
        tags={"pipeline_mode": "HISTORICAL"},
        reload_test_passed=False,
    )
    assert report.verdict == "REJECTED"
    assert report.passed is False
    assert any("rechargement externe" in r for r in report.rejection_reasons)


def test_qualification_marks_synthetic_as_demo_only(nominal_metrics):
    """Des métriques satisfaisantes obtenues sur flux synthétique donnent le statut DEMO_ONLY."""
    report = evaluate_run_qualification(
        run_id="run-synthetic",
        metrics=nominal_metrics,
        tags={"pipeline_mode": "SYNTHETIC", "synthetic": "true"},
        reload_test_passed=True,
    )
    assert report.verdict == "DEMO_ONLY"
    assert report.passed is True
    assert "DEMO_ONLY" in report.summary


# ==============================================================================
# 2. Tests de Persistance SQLite et Traçabilité des Publications
# ==============================================================================

def test_publication_manager_lifecycle_and_audit(tmp_path: Path):
    """Vérifie la création, la mise à jour par jalons et l'enregistrement d'audit."""
    db_file = tmp_path / "test_pub.db"
    manager = PublicationManager(db_path=db_file)

    # 1. Création publication
    pub_id = manager.create_publication(
        run_id="run-test-123",
        verdict="PRODUCTION_READY",
        target_alias="champion",
        regression_model_name="steam-positive-ratio",
        classification_model_name="steam-owner-range",
    )
    pub = manager.get_publication(pub_id)
    assert pub["status"] == "PENDING"
    assert pub["regression_status"] == "PENDING"

    # 2. Mise à jour de jalons
    manager.update_publication(
        pub_id,
        status="IN_PROGRESS",
        regression_version="2",
        regression_status="ALIASED",
    )
    pub = manager.get_publication(pub_id)
    assert pub["status"] == "IN_PROGRESS"
    assert pub["regression_version"] == "2"
    assert pub["regression_status"] == "ALIASED"

    # 3. Enregistrement d'événement d'audit
    evt_id = manager.record_selection_event(
        model_name="steam-positive-ratio",
        target_version="2",
        alias="champion",
        previous_version="1",
        action="PROMOTION",
        reason="Passage de qualification formelle",
        publication_id=pub_id,
    )
    events = manager.list_selection_events("steam-positive-ratio")
    assert len(events) == 1
    assert events[0]["event_id"] == evt_id
    assert events[0]["action"] == "PROMOTION"
    assert events[0]["previous_version"] == "1"


# ==============================================================================
# 3. Tests de Réconciliation après Panne et Idempotence
# ==============================================================================

def test_reconciliation_recovers_interrupted_publication_without_duplicates(tmp_path: Path):
    """Simule une publication interrompue et vérifie sa reprise sans duplication."""
    db_file = tmp_path / "test_pub.db"
    manager = PublicationManager(db_path=db_file)

    pub_id = manager.create_publication(
        run_id="run-crash-456",
        verdict="PRODUCTION_READY",
        target_alias="champion",
        regression_model_name="steam-reg-mock",
        classification_model_name="steam-cls-mock",
    )
    # Interruption simulée : regression aliased, classification non traitée
    manager.update_publication(
        pub_id,
        status="IN_PROGRESS",
        regression_version="1",
        regression_status="ALIASED",
    )

    mock_client = MagicMock()
    # Simuler qu'aucune version existante de classification n'a été créée pour ce run
    mock_client.search_model_versions.return_value = []
    created_mv = MagicMock()
    created_mv.version = "1"
    mock_client.create_model_version.return_value = created_mv

    # Réconciliation
    res = reconcile_publication(pub_id, client=mock_client, pub_manager=manager)
    assert res["status"] == "COMPLETED"
    assert res["regression_version"] == "1"
    assert res["classification_version"] == "1"

    # Vérification que la régression déjà ALIASED n'a pas été recréée
    assert mock_client.create_model_version.call_count == 1  # uniquement pour classification
    mock_client.set_registered_model_alias.assert_called_with(
        name="steam-cls-mock", alias="champion", version="1"
    )

    # Réconciliation ultérieure sur un lot déjà terminé -> idempotence
    res_again = reconcile_publication(pub_id, client=mock_client, pub_manager=manager)
    assert res_again["status"] == "COMPLETED"
    assert res_again["already_completed"] is True


# ==============================================================================
# 4. Tests de Promotion et Retour Arrière (Rollback)
# ==============================================================================

def test_model_alias_promotion_and_rollback(tmp_path: Path):
    """Vérifie la séquence complète de promotion vers champion puis de rollback avec trace d'audit."""
    db_file = tmp_path / "test_pub.db"
    manager = PublicationManager(db_path=db_file)
    mock_client = MagicMock()

    # Configuration initiale : v1 active en champion
    mv_v1 = MagicMock()
    mv_v1.version = "1"
    mock_client.get_model_version_by_alias.return_value = mv_v1

    # 1. Promotion de la version 2
    promo_res = promote_model_to_alias(
        model_name="steam-test-model",
        version="2",
        alias="champion",
        reason="Validation manuelle",
        triggered_by="test_runner",
        pub_manager=manager,
        client=mock_client,
    )
    assert promo_res["new_version"] == "2"
    assert promo_res["previous_version"] == "1"
    assert promo_res["action"] == "PROMOTION"
    mock_client.set_registered_model_alias.assert_called_with(
        name="steam-test-model", alias="champion", version="2"
    )

    # Mise à jour du mock pour refléter v2 active
    mv_v2 = MagicMock()
    mv_v2.version = "2"
    mock_client.get_model_version_by_alias.return_value = mv_v2

    # 2. Exécution du retour arrière (rollback) vers v1
    rollback_res = rollback_model_alias(
        model_name="steam-test-model",
        target_version="1",
        alias="champion",
        reason="Dérive détectée sur v2",
        triggered_by="operator_test",
        pub_manager=manager,
        client=mock_client,
    )
    assert rollback_res["restored_version"] == "1"
    assert rollback_res["replaced_version"] == "2"
    assert rollback_res["action"] == "ROLLBACK"
    mock_client.set_registered_model_alias.assert_called_with(
        name="steam-test-model", alias="champion", version="1"
    )

    # 3. Contrôle des événements d'audit
    events = manager.list_selection_events("steam-test-model")
    assert len(events) == 2
    actions = [e["action"] for e in events]
    assert "ROLLBACK" in actions
    assert "PROMOTION" in actions
