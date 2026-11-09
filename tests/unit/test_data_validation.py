"""tests/unit/test_data_validation.py - Tests unitaires de la barrière Great Expectations.

Vérifie :
1. Le passage nominal sur données conformes.
2. L'interruption immédiate sur injection d'anomalies unitaires (types, bornes, formats, sentinelles).
3. Le comportement différentié des sentinelles (historique vs nouvelles observations).
4. Le respect strict de la taille des lots PUSH (500 jeux).
5. La détection bloquante des fuites d'appids réservés.
6. L'absence absolue d'appel aux étapes avals (preprocess_data, train_models) en cas d'échec (espions).
7. L'archivage exploitable des rapports d'anomalies.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from src.contracts import REQUIRED_SNAPSHOT_COLUMNS, IngestionResult, RunContext, SnapshotRef
from src.data_validation import (
    build_historical_schema_suite,
    build_new_observations_suite,
    build_push_batch_suite,
    build_snapshot_integrity_suite,
    build_training_readiness_suite,
    validate_dataframe,
)
from src.exceptions import (
    DataIntegrityError,
    DataLeakageError,
    DataSchemaError,
    DataValidationError,
)
from src.ingestion import load_data
from src.preprocessing import preprocess_data


@pytest.fixture
def base_valid_df() -> pd.DataFrame:
    """Fixture d'un DataFrame de 10 jeux strictement conformes."""
    rows = []
    for i in range(1, 11):
        rows.append(
            {
                "appid": i,
                "name": f"Game {i}",
                "release_date": f"2020-{i:02d}-15",
                "required_age": 0 if i % 2 == 0 else 18,
                "price": float(i * 4.99),
                "dlc_count": i % 3,
                "achievements": i * 5,
                "windows": True,
                "mac": i % 2 == 0,
                "linux": False,
                "positive": 100 * i,
                "negative": 10 * i,
                "num_reviews_total": 110 * i,
                "pct_pos_total": round((100 * i) / (110 * i) * 100, 2),
                "genres": "['Action', 'Indie']" if i % 2 == 0 else "['Strategy']",
                "estimated_owners": "0 - 20000" if i <= 5 else "20000 - 50000",
            }
        )
    return pd.DataFrame(rows)[list(REQUIRED_SNAPSHOT_COLUMNS)]


# ==============================================================================
# 1. Tests Nominaux
# ==============================================================================

def test_nominal_historical_validation_passes(base_valid_df: pd.DataFrame):
    """Vérifie que des données conformes valident historical_schema_v1 sans erreur."""
    report = validate_dataframe(
        base_valid_df,
        suite_name="historical_schema_v1",
        dataset_name="nominal_test",
        raise_on_error=True,
    )
    assert report["success"] is True
    assert report["failed_checks_count"] == 0


def test_nominal_new_observations_validation_passes(base_valid_df: pd.DataFrame):
    """Vérifie que des observations sans sentinelle valident new_observations_v1."""
    report = validate_dataframe(
        base_valid_df,
        suite_name="new_observations_v1",
        dataset_name="nominal_new_obs",
        raise_on_error=True,
    )
    assert report["success"] is True
    assert report["failed_checks_count"] == 0


# ==============================================================================
# 2. Injections d'Anomalies Unitaires
# ==============================================================================

def test_anomaly_missing_column_raises_error(base_valid_df: pd.DataFrame, tmp_path: Path):
    """Une colonne obligatoire manquante doit bloquer immédiatement la validation."""
    corrupted = base_valid_df.drop(columns=["price"])
    with pytest.raises(DataValidationError) as exc_info:
        validate_dataframe(corrupted, suite_name="historical_schema_v1", output_dir=tmp_path, raise_on_error=True)
    msg = str(exc_info.value).lower()
    assert "schéma invalide" in msg or "colonnes manquantes" in msg or "contrôle(s)" in msg


def test_anomaly_duplicate_appid_raises_error(base_valid_df: pd.DataFrame, tmp_path: Path):
    """Un identifiant appid dupliqué doit être rejeté."""
    corrupted = base_valid_df.copy()
    corrupted.loc[1, "appid"] = corrupted.loc[0, "appid"]
    with pytest.raises(DataValidationError):
        validate_dataframe(corrupted, suite_name="historical_schema_v1", output_dir=tmp_path, raise_on_error=True)


def test_anomaly_negative_or_zero_appid_raises_error(base_valid_df: pd.DataFrame, tmp_path: Path):
    """Un appid négatif ou égal à zéro doit être rejeté."""
    corrupted = base_valid_df.copy()
    corrupted.loc[0, "appid"] = 0
    with pytest.raises(DataValidationError):
        validate_dataframe(corrupted, suite_name="historical_schema_v1", output_dir=tmp_path, raise_on_error=True)


def test_anomaly_textual_price_raises_error(base_valid_df: pd.DataFrame, tmp_path: Path):
    """Un prix textuel ('gratuit') viole le type et doit être rejeté."""
    corrupted = base_valid_df.copy()
    corrupted["price"] = corrupted["price"].astype(object)
    corrupted.loc[0, "price"] = "gratuit"
    with pytest.raises(DataValidationError):
        validate_dataframe(corrupted, suite_name="historical_schema_v1", output_dir=tmp_path, raise_on_error=True)


def test_anomaly_pct_pos_over_100_raises_error(base_valid_df: pd.DataFrame, tmp_path: Path):
    """Un pourcentage d'avis positif supérieur à 100 est invalide."""
    corrupted = base_valid_df.copy()
    corrupted.loc[0, "pct_pos_total"] = 125.0
    with pytest.raises(DataValidationError):
        validate_dataframe(corrupted, suite_name="historical_schema_v1", output_dir=tmp_path, raise_on_error=True)


def test_anomaly_inverted_owner_bounds_raises_error(base_valid_df: pd.DataFrame, tmp_path: Path):
    """Une tranche d'owners avec bornes inversées (ex. '50000 - 20000') doit être rejetée."""
    corrupted = base_valid_df.copy()
    corrupted.loc[0, "estimated_owners"] = "50000 - 20000"
    with pytest.raises(DataValidationError):
        validate_dataframe(corrupted, suite_name="historical_schema_v1", output_dir=tmp_path, raise_on_error=True)


def test_anomaly_non_boolean_platform_raises_error(base_valid_df: pd.DataFrame, tmp_path: Path):
    """Une valeur de plateforme non booléenne doit échouer."""
    corrupted = base_valid_df.copy()
    corrupted["windows"] = corrupted["windows"].astype(object)
    corrupted.loc[0, "windows"] = "inconnu"
    with pytest.raises(DataValidationError):
        validate_dataframe(corrupted, suite_name="historical_schema_v1", output_dir=tmp_path, raise_on_error=True)


# ==============================================================================
# 3. Comportement Différencié des Sentinelles (Historique vs Nouveaux Flux)
# ==============================================================================

def test_sentinels_handling_historical_vs_new_observations(base_valid_df: pd.DataFrame, tmp_path: Path):
    """Vérifie que les sentinelles historiques sont admises dans historical_schema_v1 mais refusées dans new_observations_v1."""
    df_sentinels = base_valid_df.copy()
    df_sentinels.loc[0, "required_age"] = -1
    df_sentinels.loc[1, "pct_pos_total"] = -1.0
    df_sentinels.loc[1, "num_reviews_total"] = -1
    df_sentinels.loc[2, "estimated_owners"] = "0 - 0"

    # 1. Autorisé dans l'historique
    hist_report = validate_dataframe(
        df_sentinels, suite_name="historical_schema_v1", output_dir=tmp_path, raise_on_error=False
    )
    assert hist_report["success"] is True

    # 2. Strictement refusé dans les nouvelles observations
    with pytest.raises(DataValidationError):
        validate_dataframe(
            df_sentinels, suite_name="new_observations_v1", output_dir=tmp_path, raise_on_error=True
        )


# ==============================================================================
# 4. Taille et Cohérence des Lots PUSH (Suite push_batch_v1)
# ==============================================================================

def test_push_batch_suite_enforces_exact_500_games(base_valid_df: pd.DataFrame, tmp_path: Path):
    """Vérifie que la suite push_batch_v1 exige rigoureusement 500 jeux uniques."""
    # Lot incomplet de 499 jeux -> rejet
    rows_499 = []
    for i in range(1, 500):
        rows_499.append(
            {
                "appid": i,
                "name": f"Push Game {i}",
                "release_date": "2024-01-01",
                "required_age": 0,
                "price": 9.99,
                "dlc_count": 0,
                "achievements": 10,
                "windows": True,
                "mac": False,
                "linux": False,
                "positive": 50,
                "negative": 5,
                "num_reviews_total": 55,
                "pct_pos_total": 90.9,
                "genres": "['Action']",
                "estimated_owners": "20000 - 50000",
            }
        )
    df_499 = pd.DataFrame(rows_499)
    with pytest.raises(DataValidationError):
        validate_dataframe(df_499, suite_name="push_batch_v1", output_dir=tmp_path, raise_on_error=True)

    # Lot complet de 500 jeux -> succès
    rows_500 = rows_499 + [
        {
            "appid": 500,
            "name": "Push Game 500",
            "release_date": "2024-01-01",
            "required_age": 0,
            "price": 9.99,
            "dlc_count": 0,
            "achievements": 10,
            "windows": True,
            "mac": False,
            "linux": False,
            "positive": 50,
            "negative": 5,
            "num_reviews_total": 55,
            "pct_pos_total": 90.9,
            "genres": "['Action']",
            "estimated_owners": "20000 - 50000",
        }
    ]
    df_500 = pd.DataFrame(rows_500)
    report_500 = validate_dataframe(df_500, suite_name="push_batch_v1", output_dir=tmp_path, raise_on_error=True)
    assert report_500["success"] is True


# ==============================================================================
# 5. Aptitude à l'Entraînement & Détection Anti-Fuite
# ==============================================================================

def test_training_readiness_rejects_monoclass_data(base_valid_df: pd.DataFrame, tmp_path: Path):
    """La suite training_readiness_v1 doit refuser un jeu ne comportant qu'une seule classe active."""
    monoclass_df = base_valid_df.copy()
    monoclass_df["estimated_owners"] = "20000 - 50000"  # Une seule classe pour tous
    with pytest.raises(DataValidationError):
        validate_dataframe(monoclass_df, suite_name="training_readiness_v1", output_dir=tmp_path, raise_on_error=True)


def test_leakage_detection_raises_data_leakage_error(base_valid_df: pd.DataFrame, tmp_path: Path):
    """La détection d'appids réservés dans le flux d'apprentissage lève DataLeakageError."""
    reserved_set = {3, 7, 999}
    with pytest.raises(DataLeakageError) as exc_info:
        validate_dataframe(
            base_valid_df,
            suite_name="training_readiness_v1",
            check_reserved=True,
            reserved_appids=reserved_set,
            output_dir=tmp_path,
            raise_on_error=True,
        )
    assert 3 in exc_info.value.sample_offending_appids or 7 in exc_info.value.sample_offending_appids


# ==============================================================================
# 6. Espions & Preuve d'Interruption Totale
# ==============================================================================

def test_pipeline_halts_and_never_calls_preprocessing_or_training_on_invalid_data(tmp_path: Path):
    """Prouve par espions qu'en cas d'anomalie, preprocess_data et train_models ne sont JAMAIS appelés."""
    # Création d'un snapshot corrompu
    corrupted_data = pd.DataFrame(
        [
            {"appid": 1, "name": "Valid 1", "price": 10.0},
            {"appid": 1, "name": "Duplicate Appid", "price": 10.0},  # Duplicat
        ]
    )
    corrupted_csv = tmp_path / "corrupted_snapshot.csv"
    corrupted_data.to_csv(corrupted_csv, index=False)

    test_config = {
        "snapshot_path": str(corrupted_csv),
        "validation_suite": "historical_schema_v1",
    }
    config_file = tmp_path / "test_config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(test_config, f)

    with patch("src.preprocessing.preprocess_data") as spy_preprocess, patch(
        "src.training.train_models"
    ) as spy_training:
        with pytest.raises(DataValidationError):
            load_data(config_file)

        # Les fonctions mères avals ne doivent avoir été appelées 0 fois
        spy_preprocess.assert_not_called()
        spy_training.assert_not_called()


def test_failure_report_archived_with_offending_appids(tmp_path: Path, base_valid_df: pd.DataFrame):
    """Vérifie qu'un échec de validation archive un rapport JSON détaillé contenant les appids fautifs."""
    corrupted = base_valid_df.copy()
    corrupted.loc[2, "pct_pos_total"] = 150.0  # Appid 3
    corrupted.loc[4, "pct_pos_total"] = 200.0  # Appid 5

    with pytest.raises(DataValidationError) as exc_info:
        validate_dataframe(
            corrupted,
            suite_name="historical_schema_v1",
            output_dir=tmp_path,
            raise_on_error=True,
        )

    # Vérification du rapport généré
    archived_reports = list(tmp_path.glob("validation_failure_*.json"))
    assert len(archived_reports) == 1
    with open(archived_reports[0], "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["success"] is False
    assert data["failed_checks_count"] >= 1
    assert 3 in data["sample_offending_appids"] or 5 in data["sample_offending_appids"]
    assert exc_info.value.report_path == str(archived_reports[0])
