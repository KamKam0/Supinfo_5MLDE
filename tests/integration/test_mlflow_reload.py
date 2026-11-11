"""tests/integration/test_mlflow_reload.py - Tests de rechargement externe et intégrité MLflow.

Vérifie :
1. La structure du run MLflow (paramètres, métriques, baselines, tags, empreinte SHA-256).
2. L'absence absolue de variables cibles dans la signature d'inférence (étanchéité anti-fuite).
3. Le rechargement des pipelines scikit-learn (transformeur de features inclus) avec fidélité numérique à 1e-10.
4. Le rechargement fonctionnel et l'inférence dans un sous-processus Python complètement isolé.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import mlflow
from mlflow.models import get_model_info
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.contracts import PreparedData, RunContext, SnapshotRef, TaskDataset
from src.feature_builder import SteamFeatureBuilder
from src.settings import settings
from src.tracking.mlflow_tracking import (
    CLASSIFICATION_MODEL_NAME,
    REGRESSION_MODEL_NAME,
    init_mlflow,
    log_training_run,
)
from src.tracking.registry import load_model, predict_pipeline
from src.training import TrainedModelsBundle, create_classification_pipeline, create_regression_pipeline


@pytest.fixture(scope="module")
def mlflow_run_fixture():
    """Crée un run d'entraînement synthétique complet et le journalise dans MLflow."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", settings.MLFLOW_TRACKING_URI)
    client, exp_id = init_mlflow(
        tracking_uri=tracking_uri,
        experiment_name="steam-mlops-test-suite",
    )

    # 1. Génération de données d'apprentissage synthétiques
    n_samples = 120
    genres_pool = ["['Action']", "['Indie']", "['Adventure']", "['RPG']", "['Strategy']"]
    owners_pool = ["20000 - 50000", "50000 - 100000", "100000 - 200000", "200000 - 500000"]

    rows = []
    for i in range(1, n_samples + 1):
        pos = 100 + i * 5
        neg = 10 + (i % 10)
        tot = pos + neg
        rows.append({
            "appid": 3000000 + i,
            "name": f"Game {i}",
            "release_date": "2023-05-10",
            "required_age": 0 if i % 2 == 0 else 16,
            "price": float((i % 15) * 1.99),
            "dlc_count": i % 3,
            "windows": True,
            "mac": i % 2 == 0,
            "linux": False,
            "achievements": (i % 20) * 2,
            "genres": genres_pool[i % len(genres_pool)],
            "positive": pos,
            "negative": neg,
            "num_reviews_total": tot,
            "pct_pos_total": round(100.0 * pos / float(tot), 2),
            "estimated_owners": owners_pool[i % len(owners_pool)],
        })
    df = pd.DataFrame(rows)

    appids = df["appid"].tolist()
    train_ids = appids[:80]
    val_ids = appids[80:100]
    test_ids = appids[100:]

    # 2. Entraînement de pipelines miniatures
    reg_pipe = create_regression_pipeline(n_estimators=5, max_depth=4)
    cls_pipe = create_classification_pipeline(n_estimators=5, max_depth=4)

    features = list(settings.ALLOWED_INFERENCE_FEATURES)
    train_df = df[df["appid"].isin(set(train_ids))]

    reg_pipe.fit(train_df[features], train_df[settings.TARGET_REGRESSION].values)
    cls_pipe.fit(train_df[features], train_df[settings.TARGET_CLASSIFICATION].values)

    bundle = TrainedModelsBundle(
        regression_pipeline=reg_pipe,
        regression_baseline=reg_pipe,
        classification_pipeline=cls_pipe,
        classification_baseline=cls_pipe,
        training_durations={"regression_seconds": 0.15, "classification_seconds": 0.20, "total_training_seconds": 0.35},
    )

    # 3. Métriques synthétiques
    eval_metrics = {
        "regression": {
            "train": {"candidate": {"mae": 5.2, "rmse": 6.8, "r2": 0.45}},
            "validation": {
                "candidate": {"mae": 6.1, "rmse": 7.9, "r2": 0.38},
                "baseline": {"mae": 10.5, "rmse": 12.1, "r2": -0.05},
            },
            "test": {"candidate": {"mae": 6.3, "rmse": 8.0, "r2": 0.36}},
        },
        "classification": {
            "train": {"candidate": {"accuracy": 0.85, "f1_macro": 0.82}},
            "validation": {
                "candidate": {"accuracy": 0.78, "f1_macro": 0.75},
                "baseline": {"accuracy": 0.35, "f1_macro": 0.20},
            },
            "test": {"candidate": {"accuracy": 0.76, "f1_macro": 0.73}},
        },
        "training_durations_sec": bundle.training_durations,
    }

    context = RunContext(
        run_id="test-run-mlflow-l08",
        timestamp="2026-11-16T12:00:00Z",
        config_path="config/training.historical.json",
        mode="HISTORICAL",
        gx_validated=True,
    )
    snapshot = SnapshotRef(
        path="data/snapshots/historical_projection.csv",
        sha256="7eec2ff4e6f1389bfb74308c8de1206fee2a94f51120895c2cf7fce84146312f",
        row_count=n_samples,
        column_count=16,
    )
    prepared_data = PreparedData(
        context=context,
        regression_data=TaskDataset(
            task_name="regression",
            train_appids=train_ids,
            val_appids=val_ids,
            test_appids=test_ids,
            feature_names=features,
            target_name=settings.TARGET_REGRESSION,
        ),
        classification_data=TaskDataset(
            task_name="classification",
            train_appids=train_ids,
            val_appids=val_ids,
            test_appids=test_ids,
            feature_names=features,
            target_name=settings.TARGET_CLASSIFICATION,
        ),
    )

    log_info = log_training_run(
        models_bundle=bundle,
        prepared_data=prepared_data,
        eval_metrics=eval_metrics,
        sample_df=df,
        snapshot=snapshot,
        tracking_uri=tracking_uri,
        experiment_name="steam-mlops-test-suite",
        run_name="test-l08-mlflow-reload",
        register_models=True,
        model_name_suffix="-test",
    )

    return {
        "log_info": log_info,
        "client": client,
        "bundle": bundle,
        "df": df,
        "features": features,
    }


def test_mlflow_run_structure_and_metrics(mlflow_run_fixture):
    """Vérifie la présence et la conformité des paramètres, métriques, baselines et tags dans MLflow."""
    fixture = mlflow_run_fixture
    client: MlflowClient = fixture["client"]
    run_id = fixture["log_info"]["mlflow_run_id"]

    run = client.get_run(run_id)
    params = run.data.params
    metrics = run.data.metrics
    tags = run.data.tags

    # 1. Contrôle des paramètres
    assert "split.train_ratio" in params
    assert params["features.top_genres_k"] == "15"
    assert params["regression.estimator"] == "RandomForestRegressor"
    assert params["classification.estimator"] == "RandomForestClassifier"

    # 2. Contrôle des métriques et des baselines
    assert "regression.val.mae" in metrics
    assert "regression.val.baseline_mae" in metrics
    assert metrics["regression.val.mae"] < metrics["regression.val.baseline_mae"]
    assert "regression.val.improvement_mae_pct" in metrics

    assert "classification.val.accuracy" in metrics
    assert "classification.val.baseline_accuracy" in metrics
    assert metrics["classification.val.accuracy"] > metrics["classification.val.baseline_accuracy"]
    assert "classification.val.improvement_acc_pct" in metrics

    assert "duration.total_train_sec" in metrics

    # 3. Contrôle des tags de traçabilité
    assert tags["snapshot_sha256"] == "7eec2ff4e6f1389bfb74308c8de1206fee2a94f51120895c2cf7fce84146312f"
    assert "git_commit" in tags
    assert "python_version" in tags
    assert tags["framework"] == "scikit-learn"


def test_mlflow_model_signatures_have_no_target_columns(mlflow_run_fixture):
    """Vérifie que les signatures d'inférence n'exigent et n'acceptent AUCUNE variable cible ni fuite."""
    fixture = mlflow_run_fixture
    reg_uri = fixture["log_info"]["regression_model_uri"]
    cls_uri = fixture["log_info"]["classification_model_uri"]

    for uri in [reg_uri, cls_uri]:
        model_info = get_model_info(uri)
        sig = model_info.signature
        assert sig is not None, f"Signature manquante pour {uri}"

        # Extraction des noms de colonnes en entrée
        input_cols = [c.name for c in sig.inputs.inputs]

        # Aucune variable cible ni de fuite ne doit être présente
        for forbidden in settings.LEAKAGE_COLUMNS:
            assert forbidden not in input_cols, f"Variable fuyante/cible '{forbidden}' détectée dans la signature de {uri} !"

        # Toutes les caractéristiques autorisées doivent être déclarées
        for allowed in settings.ALLOWED_INFERENCE_FEATURES:
            assert allowed in input_cols, f"Caractéristique légitime '{allowed}' absente de la signature de {uri}."


def test_mlflow_model_reload_and_exact_prediction(mlflow_run_fixture):
    """Vérifie le rechargement via l'URI MLflow et la reproduction exacte des prédictions (tolérance 1e-10)."""
    fixture = mlflow_run_fixture
    reg_uri = fixture["log_info"]["regression_model_uri"]
    cls_uri = fixture["log_info"]["classification_model_uri"]
    df = fixture["df"]
    features = fixture["features"]

    # 1. Rechargement des modèles
    reloaded_reg = load_model(reg_uri)
    reloaded_cls = load_model(cls_uri)

    assert isinstance(reloaded_reg, Pipeline)
    assert isinstance(reloaded_cls, Pipeline)
    assert "features" in reloaded_reg.named_steps
    assert isinstance(reloaded_reg.named_steps["features"], SteamFeatureBuilder)

    # 2. Préparation du jeu de test (strictement dénué de cibles)
    test_X = df[features].head(20).copy()

    # 3. Comparaison avec les modèles d'origine en mémoire
    original_bundle: TrainedModelsBundle = fixture["bundle"]

    orig_reg_pred = original_bundle.regression_pipeline.predict(test_X)
    reloaded_reg_pred = predict_pipeline(reloaded_reg, test_X)
    np.testing.assert_allclose(reloaded_reg_pred, orig_reg_pred, rtol=1e-10, atol=1e-10)

    orig_cls_pred = original_bundle.classification_pipeline.predict(test_X)
    reloaded_cls_pred = predict_pipeline(reloaded_cls, test_X)
    assert list(reloaded_cls_pred) == list(orig_cls_pred)


def test_mlflow_model_reload_in_isolated_subprocess(mlflow_run_fixture):
    """Prouve formellement le rechargement et l'inférence dans un sous-processus Python complètement neuf."""
    fixture = mlflow_run_fixture
    reg_uri = fixture["log_info"]["regression_model_uri"]
    cls_uri = fixture["log_info"]["classification_model_uri"]
    tracking_uri = fixture["log_info"]["tracking_uri"]

    # Script autonome exécuté dans un interpréteur vierge
    isolated_script = f"""
import sys
import pandas as pd
import numpy as np
from src.tracking.registry import load_model, predict_pipeline

tracking_uri = "{tracking_uri}"
reg_uri = "{reg_uri}"
cls_uri = "{cls_uri}"

# Rechargement des deux pipelines
reg_model = load_model(reg_uri, tracking_uri=tracking_uri)
cls_model = load_model(cls_uri, tracking_uri=tracking_uri)

# Jeux de données d'inférence avec cas limites (genres inconnus, valeurs manquantes)
games = pd.DataFrame([
    {{
        "price": 29.99,
        "required_age": 18,
        "dlc_count": 2,
        "achievements": 50,
        "windows": True,
        "mac": False,
        "linux": False,
        "genres": "['Action', 'CyberpunkSpecialGenre']",
        "release_date": "2024-11-01",
    }},
    {{
        "price": 0.0,
        "required_age": 0,
        "dlc_count": 0,
        "achievements": 0,
        "windows": True,
        "mac": True,
        "linux": True,
        "genres": "['Indie']",
        "release_date": None,
    }}
])

# Exécution de l'inférence
reg_preds = predict_pipeline(reg_model, games)
cls_preds = predict_pipeline(cls_model, games)

# Vérifications
assert len(reg_preds) == 2, f"Attendu 2 prédictions régression, obtenu {{len(reg_preds)}}"
assert len(cls_preds) == 2, f"Attendu 2 prédictions classification, obtenu {{len(cls_preds)}}"
assert all(0.0 <= p <= 100.0 for p in reg_preds), f"Prédictions régression hors limites : {{reg_preds}}"
assert all(isinstance(c, str) and "-" in c for c in cls_preds), f"Classes de possession invalides : {{cls_preds}}"

print("SUBPROCESS_SUCCESS")
"""

    res = subprocess.run(
        [sys.executable, "-c", isolated_script],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
        cwd=str(settings.PROJECT_ROOT),
    )

    assert res.returncode == 0, f"Erreur dans le sous-processus :\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
    assert "SUBPROCESS_SUCCESS" in res.stdout
