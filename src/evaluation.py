"""src/evaluation.py - Fonction mère 4 : evaluate_models.

Évalue les modèles candidats et leurs baselines sur Train, Validation et Test,
sauvegarde les métriques détaillées et sérialise les pipelines joblib.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

from src.contracts import PreparedData
from src.settings import settings
from src.training import TrainedModelsBundle


def evaluate_models(
    models_bundle: TrainedModelsBundle,
    prepared_data: PreparedData,
    full_df: pd.DataFrame,
    artifacts_dir: Path | str = "artifacts",
) -> Dict[str, Any]:
    """Quatrième fonction mère : évaluation multi-corpus et sérialisation des modèles."""
    art_dir = Path(artifacts_dir)
    art_dir.mkdir(parents=True, exist_ok=True)
    features = list(settings.ALLOWED_INFERENCE_FEATURES)

    # --------------------------------------------------------------------------
    # 1. Évaluation Régression
    # --------------------------------------------------------------------------
    reg_data = prepared_data.regression_data
    reg_pipe = models_bundle.regression_pipeline
    reg_base = models_bundle.regression_baseline

    def eval_reg_split(appids: list[int]) -> Dict[str, float]:
        sub_df = full_df[full_df["appid"].isin(set(appids))]
        X = sub_df[features]
        y_true = sub_df[settings.TARGET_REGRESSION].values

        y_pred = reg_pipe.predict(X)
        y_base = reg_base.predict(X)

        return {
            "samples": len(y_true),
            "candidate": {
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "rmse": float(root_mean_squared_error(y_true, y_pred)),
                "r2": float(r2_score(y_true, y_pred)),
            },
            "baseline": {
                "mae": float(mean_absolute_error(y_true, y_base)),
                "rmse": float(root_mean_squared_error(y_true, y_base)),
                "r2": float(r2_score(y_true, y_base)),
            },
        }

    reg_metrics = {
        "train": eval_reg_split(reg_data.train_appids),
        "validation": eval_reg_split(reg_data.val_appids),
        "test": eval_reg_split(reg_data.test_appids),
    }

    # --------------------------------------------------------------------------
    # 2. Évaluation Classification
    # --------------------------------------------------------------------------
    cls_data = prepared_data.classification_data
    cls_pipe = models_bundle.classification_pipeline
    cls_base = models_bundle.classification_baseline

    def eval_cls_split(appids: list[int]) -> Dict[str, float]:
        sub_df = full_df[full_df["appid"].isin(set(appids))]
        X = sub_df[features]
        y_true = sub_df[settings.TARGET_CLASSIFICATION].values

        y_pred = cls_pipe.predict(X)
        y_base = cls_base.predict(X)

        return {
            "samples": len(y_true),
            "candidate": {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            },
            "baseline": {
                "accuracy": float(accuracy_score(y_true, y_base)),
                "f1_macro": float(f1_score(y_true, y_base, average="macro", zero_division=0)),
            },
        }

    cls_metrics = {
        "train": eval_cls_split(cls_data.train_appids),
        "validation": eval_cls_split(cls_data.val_appids),
        "test": eval_cls_split(cls_data.test_appids),
    }

    # --------------------------------------------------------------------------
    # 3. Sérialisation des Modèles et Manifeste Local
    # --------------------------------------------------------------------------
    reg_path = art_dir / "regression.joblib"
    cls_path = art_dir / "classification.joblib"

    joblib.dump(reg_pipe, reg_path)
    joblib.dump(cls_pipe, cls_path)

    metrics_payload = {
        "run_id": prepared_data.context.run_id,
        "regression": reg_metrics,
        "classification": cls_metrics,
        "training_durations_sec": models_bundle.training_durations,
    }

    metrics_path = art_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    # Manifeste local des pipelines sérialisés (mis à jour par scripts/select_models.py)
    state_dir = art_dir.parent / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    manifest_local = {
        "run_id": prepared_data.context.run_id,
        "regression_model_path": str(reg_path),
        "classification_model_path": str(cls_path),
        "metrics_path": str(metrics_path),
        "created_at": prepared_data.context.timestamp,
    }
    with open(state_dir / "local-models.json", "w", encoding="utf-8") as f:
        json.dump(manifest_local, f, indent=2)

    print("-" * 70)
    print("RÉSULTATS DE VALIDATION (QUALIFICATION CANDIDATS VS BASELINES) :")
    print(
        f"  Régression MAE (Validation)      : {reg_metrics['validation']['candidate']['mae']:.2f}% "
        f"(vs Baseline {reg_metrics['validation']['baseline']['mae']:.2f}%)"
    )
    print(
        f"  Régression R² (Validation)       : {reg_metrics['validation']['candidate']['r2']:.4f} "
        f"(vs Baseline {reg_metrics['validation']['baseline']['r2']:.4f})"
    )
    print(
        f"  Classification Accuracy (Val)    : {cls_metrics['validation']['candidate']['accuracy']:.2%} "
        f"(vs Baseline {cls_metrics['validation']['baseline']['accuracy']:.2%})"
    )
    print(
        f"  Classification F1-Macro (Val)    : {cls_metrics['validation']['candidate']['f1_macro']:.4f} "
        f"(vs Baseline {cls_metrics['validation']['baseline']['f1_macro']:.4f})"
    )
    print(f"Modèles sérialisés dans : {art_dir}")
    print("-" * 70)

    return metrics_payload
