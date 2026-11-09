"""src/training.py - Fonction mère 3 : train_models.

Entraîne les pipelines complets scikit-learn (transformeur + imputeur + estimateur)
et leurs baselines naïves respectives pour la régression et la classification.
"""

from __future__ import annotations

import time
from typing import Any, Dict
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.contracts import PreparedData
from src.feature_builder import SteamFeatureBuilder
from src.settings import settings


class TrainedModelsBundle:
    """Conteneur des pipelines entraînés et de leurs métadonnées de construction."""

    def __init__(
        self,
        regression_pipeline: Pipeline,
        regression_baseline: Pipeline,
        classification_pipeline: Pipeline,
        classification_baseline: Pipeline,
        training_durations: Dict[str, float],
    ):
        self.regression_pipeline = regression_pipeline
        self.regression_baseline = regression_baseline
        self.classification_pipeline = classification_pipeline
        self.classification_baseline = classification_baseline
        self.training_durations = training_durations


def create_regression_pipeline(n_estimators: int = 40, max_depth: int = 12) -> Pipeline:
    """Construit le pipeline complet de régression."""
    return Pipeline([
        ("features", SteamFeatureBuilder(top_genres_k=15)),
        ("imputer", SimpleImputer(strategy="median")),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=settings.RANDOM_SEED,
                n_jobs=1,
            ),
        ),
    ])


def create_regression_baseline() -> Pipeline:
    """Construit la baseline naïve moyenne pour la régression."""
    return Pipeline([
        ("features", SteamFeatureBuilder(top_genres_k=15)),
        ("imputer", SimpleImputer(strategy="median")),
        ("baseline", DummyRegressor(strategy="mean")),
    ])


def create_classification_pipeline(n_estimators: int = 40, max_depth: int = 12) -> Pipeline:
    """Construit le pipeline complet de classification."""
    return Pipeline([
        ("features", SteamFeatureBuilder(top_genres_k=15)),
        ("imputer", SimpleImputer(strategy="median")),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=settings.RANDOM_SEED,
                n_jobs=1,
            ),
        ),
    ])


def create_classification_baseline() -> Pipeline:
    """Construit la baseline naïve classe majoritaire pour la classification."""
    return Pipeline([
        ("features", SteamFeatureBuilder(top_genres_k=15)),
        ("imputer", SimpleImputer(strategy="median")),
        ("baseline", DummyClassifier(strategy="most_frequent")),
    ])


def train_models(prepared_data: PreparedData, full_df: pd.DataFrame) -> TrainedModelsBundle:
    """Troisième fonction mère : ajustement des modèles et des baselines sur Train."""
    durations: Dict[str, float] = {}
    features = list(settings.ALLOWED_INFERENCE_FEATURES)

    # --------------------------------------------------------------------------
    # 1. Entraînement Régression
    # --------------------------------------------------------------------------
    reg_train_ids = set(prepared_data.regression_data.train_appids)
    reg_train_df = full_df[full_df["appid"].isin(reg_train_ids)]
    X_reg_train = reg_train_df[features]
    y_reg_train = reg_train_df[settings.TARGET_REGRESSION].values

    print(f"[TRAINING] Entraînement Régression (Train = {len(X_reg_train):,} exemples)...")
    reg_pipe = create_regression_pipeline()
    t0 = time.time()
    reg_pipe.fit(X_reg_train, y_reg_train)
    durations["regression_candidate"] = round(time.time() - t0, 3)

    reg_base = create_regression_baseline()
    t0 = time.time()
    reg_base.fit(X_reg_train, y_reg_train)
    durations["regression_baseline"] = round(time.time() - t0, 3)

    # --------------------------------------------------------------------------
    # 2. Entraînement Classification
    # --------------------------------------------------------------------------
    cls_train_ids = set(prepared_data.classification_data.train_appids)
    cls_train_df = full_df[full_df["appid"].isin(cls_train_ids)]
    X_cls_train = cls_train_df[features]
    y_cls_train = cls_train_df[settings.TARGET_CLASSIFICATION].values

    print(f"[TRAINING] Entraînement Classification (Train = {len(X_cls_train):,} exemples)...")
    cls_pipe = create_classification_pipeline()
    t0 = time.time()
    cls_pipe.fit(X_cls_train, y_cls_train)
    durations["classification_candidate"] = round(time.time() - t0, 3)

    cls_base = create_classification_baseline()
    t0 = time.time()
    cls_base.fit(X_cls_train, y_cls_train)
    durations["classification_baseline"] = round(time.time() - t0, 3)

    print(
        f"[TRAINING] Entraînements terminés en "
        f"{sum(durations.values()):.2f}s (Reg: {durations['regression_candidate']}s, "
        f"Cls: {durations['classification_candidate']}s)"
    )

    return TrainedModelsBundle(
        regression_pipeline=reg_pipe,
        regression_baseline=reg_base,
        classification_pipeline=cls_pipe,
        classification_baseline=cls_base,
        training_durations=durations,
    )
