"""tests/test_data_projection.py - Tests de validation de la projection et de la qualité des données."""

import hashlib
import json
from pathlib import Path
import pandas as pd
import pytest


@pytest.fixture
def project_root():
    return Path(__file__).resolve().parent.parent


def test_source_manifest_valid(project_root):
    manifest_path = project_root / "data" / "source_manifest.json"
    assert manifest_path.exists(), "Le fichier data/source_manifest.json doit exister."
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["source_sha256"] == "1ff3b4fa2eeeb7faf1d9c37abb63b9522813532fbf68f4c4a236d5a466688db7"
    assert manifest["projection_sha256"] == "7eec2ff4e6f1389bfb74308c8de1206fee2a94f51120895c2cf7fce84146312f"
    assert manifest["row_count"] == 94948
    assert manifest["column_count"] == 16


def test_projection_structure_and_counts(project_root):
    csv_path = project_root / "data" / "snapshots" / "historical_projection.csv"
    assert csv_path.exists(), "La projection historique doit exister dans data/snapshots/"

    # Vérification SHA-256
    sha = hashlib.sha256(csv_path.read_bytes()).hexdigest().lower()
    assert sha == "7eec2ff4e6f1389bfb74308c8de1206fee2a94f51120895c2cf7fce84146312f"

    # Vérification contenu
    df = pd.read_csv(csv_path)
    assert len(df) == 94948
    assert df["appid"].is_unique

    # Vérification des 16 colonnes requises
    expected_cols = [
        "appid", "name", "release_date", "required_age", "price", "dlc_count",
        "windows", "mac", "linux", "achievements", "genres",
        "positive", "negative", "pct_pos_total", "num_reviews_total", "estimated_owners"
    ]
    assert list(df.columns) == expected_cols


def test_eligibility_counts(project_root):
    csv_path = project_root / "data" / "snapshots" / "historical_projection.csv"
    df = pd.read_csv(csv_path)

    # Régression : pct_pos_total != -1 ET num_reviews_total >= 10
    reg_mask = (df["pct_pos_total"] != -1) & (df["num_reviews_total"] >= 10)
    assert reg_mask.sum() == 55373, "La population éligible pour la régression doit être exactement 55 373."

    # Classification : estimated_owners != '0 - 0' et classes >= 5
    cls_filtered = df[df["estimated_owners"] != "0 - 0"]
    counts = cls_filtered["estimated_owners"].value_counts()
    valid_classes = counts[counts >= 5].index
    cls_eligible = cls_filtered[cls_filtered["estimated_owners"].isin(valid_classes)]
    assert len(cls_eligible) == 81290, "La population éligible pour la classification doit être exactement 81 290."
    assert cls_eligible["estimated_owners"].nunique() == 12, "La classification doit comporter exactement 12 classes."


def test_fixtures_availability(project_root):
    inference_example = project_root / "data" / "examples" / "inference.csv"
    assert inference_example.exists(), "Le fichier data/examples/inference.csv doit exister."
    df_inf = pd.read_csv(inference_example)
    assert len(df_inf) >= 3
    # Les cibles et fuites ne doivent pas figurer dans le fichier d'inférence
    assert "pct_pos_total" not in df_inf.columns
    assert "estimated_owners" not in df_inf.columns
    assert "num_reviews_total" not in df_inf.columns

    synthetic_fixtures = project_root / "tests" / "fixtures" / "synthetic_games.json"
    assert synthetic_fixtures.exists(), "Le fichier tests/fixtures/synthetic_games.json doit exister."
    synth = json.loads(synthetic_fixtures.read_text(encoding="utf-8"))
    assert "nominal_games" in synth
    assert "invalid_anomalies" in synth
