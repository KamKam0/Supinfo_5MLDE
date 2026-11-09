"""tests/unit/test_training_pipeline.py - Tests unitaires de la chaîne d'entraînement et des modèles."""

import tempfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from src.contracts import InferenceInputGame
from src.training import (
    create_classification_baseline,
    create_classification_pipeline,
    create_regression_baseline,
    create_regression_pipeline,
)


@pytest.fixture
def synthetic_training_data():
    """Jeu de données synthétique pour les tests unitaires d'entraînement rapide."""
    data = [
        {
            "price": 0.0,
            "required_age": 0,
            "dlc_count": 1,
            "achievements": 5,
            "windows": True,
            "mac": False,
            "linux": True,
            "genres": "['Action', 'Free To Play']",
            "release_date": "2015-05-10",
            "pct_pos_total": 85.0,
            "estimated_owners": "50000 - 100000",
        },
        {
            "price": 19.99,
            "required_age": 18,
            "dlc_count": 0,
            "achievements": 30,
            "windows": True,
            "mac": True,
            "linux": False,
            "genres": "['RPG', 'Action']",
            "release_date": "2019-11-20",
            "pct_pos_total": 78.5,
            "estimated_owners": "20000 - 50000",
        },
        {
            "price": 9.99,
            "required_age": 0,
            "dlc_count": 3,
            "achievements": 12,
            "windows": True,
            "mac": False,
            "linux": False,
            "genres": "['Strategy', 'Indie']",
            "release_date": "2021-02-14",
            "pct_pos_total": 92.0,
            "estimated_owners": "20000 - 50000",
        },
        {
            "price": 4.99,
            "required_age": 0,
            "dlc_count": 0,
            "achievements": 0,
            "windows": True,
            "mac": False,
            "linux": False,
            "genres": "['Casual']",
            "release_date": "2022-07-01",
            "pct_pos_total": 60.0,
            "estimated_owners": "0 - 20000",
        },
    ] * 5  # 20 lignes
    return pd.DataFrame(data)


def test_regression_pipeline_train_predict(synthetic_training_data):
    features = [
        "price", "required_age", "dlc_count", "achievements",
        "windows", "mac", "linux", "genres", "release_date"
    ]
    X = synthetic_training_data[features]
    y = synthetic_training_data["pct_pos_total"].values

    pipe = create_regression_pipeline(n_estimators=5, max_depth=4)
    pipe.fit(X, y)

    preds = pipe.predict(X)
    assert len(preds) == len(X)
    assert (preds >= 0.0).all() and (preds <= 100.0).all()

    # Test baseline
    base = create_regression_baseline()
    base.fit(X, y)
    base_preds = base.predict(X)
    assert len(base_preds) == len(X)


def test_classification_pipeline_train_predict(synthetic_training_data):
    features = [
        "price", "required_age", "dlc_count", "achievements",
        "windows", "mac", "linux", "genres", "release_date"
    ]
    X = synthetic_training_data[features]
    y = synthetic_training_data["estimated_owners"].values

    pipe = create_classification_pipeline(n_estimators=5, max_depth=4)
    pipe.fit(X, y)

    preds = pipe.predict(X)
    assert len(preds) == len(X)
    # Les prédictions sont des classes connues
    assert set(preds).issubset(set(y))


def test_pipeline_serialization_and_reload(synthetic_training_data):
    features = [
        "price", "required_age", "dlc_count", "achievements",
        "windows", "mac", "linux", "genres", "release_date"
    ]
    X = synthetic_training_data[features]
    y = synthetic_training_data["pct_pos_total"].values

    pipe = create_regression_pipeline(n_estimators=5, max_depth=4)
    pipe.fit(X, y)
    orig_preds = pipe.predict(X)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / "test_reg.joblib"
        joblib.dump(pipe, model_path)

        reloaded = joblib.load(model_path)
        new_preds = reloaded.predict(X)

        np.testing.assert_array_almost_equal(orig_preds, new_preds)
