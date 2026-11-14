"""src/settings.py - Configuration centralisée et paramètres applicatifs typés."""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar, List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Paramètres applicatifs, scientifiques et opérationnels de la plateforme MLOps."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Répertoires fondamentaux
    PROJECT_ROOT: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    # Ports des services
    POSTGRES_PORT: int = Field(default=5432, alias="POSTGRES_PORT")
    MLFLOW_PORT: int = Field(default=7281, alias="MLFLOW_PORT")
    PREFECT_PORT: int = Field(default=4200, alias="PREFECT_PORT")
    GRAFANA_PORT: int = Field(default=3001, alias="GRAFANA_PORT")
    API_INFERENCE_PORT: int = Field(default=8001, alias="API_INFERENCE_PORT")
    API_COLLECTOR_PORT: int = Field(default=8000, alias="API_COLLECTOR_PORT")

    # Identifiants & Base de données
    POSTGRES_DB: str = Field(default="steam_api", alias="POSTGRES_DB")
    POSTGRES_USER: str = Field(default="postgres", alias="POSTGRES_USER")
    POSTGRES_PASSWORD: str = Field(default="5mlde-local-postgres", alias="POSTGRES_PASSWORD")
    API_DB_PASSWORD: str = Field(default="5mlde-local-api", alias="API_DB_PASSWORD")
    OPS_DB_PASSWORD: str = Field(default="5mlde-local-ops", alias="OPS_DB_PASSWORD")
    GRAFANA_DB_PASSWORD: str = Field(default="5mlde-local-grafana", alias="GRAFANA_DB_PASSWORD")

    # URIs d'accès
    MLFLOW_TRACKING_URI: str = Field(default="http://127.0.0.1:7281", alias="MLFLOW_TRACKING_URI")
    MLFLOW_REGISTRY_URI: str = Field(default="http://127.0.0.1:7281", alias="MLFLOW_REGISTRY_URI")
    PREFECT_API_URL: str = Field(default="http://127.0.0.1:4200/api", alias="PREFECT_API_URL")
    DATABASE_URL: str = Field(
        default="postgresql://api_writer:5mlde-local-api@127.0.0.1:5432/steam_api",
        alias="DATABASE_URL",
    )
    OPS_DATABASE_URL: str = Field(
        default="postgresql://ops_writer:5mlde-local-ops@127.0.0.1:5432/steam_api",
        alias="OPS_DATABASE_URL",
    )

    # Constantes Scientifiques & Anti-Fuite
    RANDOM_SEED: int = 42
    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15

    # Règles d'éligibilité Métier
    REGRESSION_MIN_REVIEWS: int = 10
    REGRESSION_MIN_PCT: float = 0.0
    REGRESSION_MAX_PCT: float = 100.0
    CLASSIFICATION_EXCLUDED_CLASS: str = "0 - 0"
    CLASSIFICATION_MIN_CLASS_COUNT: int = 5

    # Listes de champs autorisés et réservés
    ID_COLUMN: str = "appid"
    NAME_COLUMN: str = "name"
    TARGET_REGRESSION: str = "pct_pos_total"
    TARGET_CLASSIFICATION: str = "estimated_owners"

    NUMERIC_FEATURES: ClassVar[List[str]] = ["price", "required_age", "dlc_count", "achievements"]
    BOOLEAN_FEATURES: ClassVar[List[str]] = ["windows", "mac", "linux"]
    CATEGORICAL_FEATURES: ClassVar[List[str]] = ["genres"]
    TEMPORAL_FEATURES: ClassVar[List[str]] = ["release_date"]

    ALLOWED_INFERENCE_FEATURES: ClassVar[List[str]] = [
        "price",
        "required_age",
        "dlc_count",
        "achievements",
        "windows",
        "mac",
        "linux",
        "genres",
        "release_date",
    ]

    LEAKAGE_COLUMNS: ClassVar[List[str]] = [
        "positive",
        "negative",
        "num_reviews_total",
        "pct_pos_total",
        "estimated_owners",
    ]


settings = AppSettings()
