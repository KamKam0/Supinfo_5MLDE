"""src/contracts.py - Modèles Pydantic et contrats de données pour le cycle MLOps.

Définit :
- Les schémas de données historiques et nouvelles observations.
- Le schéma strict d'inférence (rejet absolu de toute variable cible ou fuite).
- Le schéma de feedback idempotent conforme aux contraintes SQL PostgreSQL.
- Les structures typées échangées entre tâches d'orchestration (RunContext, SnapshotRef, etc.).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Union
from uuid import UUID, uuid4
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ==============================================================================
# 1. Contrats d'Ingestion & Données Métier
# ==============================================================================

REQUIRED_SNAPSHOT_COLUMNS: List[str] = [
    "appid",
    "name",
    "release_date",
    "required_age",
    "price",
    "dlc_count",
    "achievements",
    "windows",
    "mac",
    "linux",
    "positive",
    "negative",
    "num_reviews_total",
    "pct_pos_total",
    "genres",
    "estimated_owners",
]

OWNER_RANGE_REGEX = re.compile(r"^(\d{1,20})\s*-\s*(\d{1,20})$")


def validate_owner_range_str(value: str) -> str:
    """Valide qu'une tranche de propriétaires est composée de deux entiers croissants."""
    value = value.strip()
    match = OWNER_RANGE_REGEX.match(value)
    if not match:
        raise ValueError(f"Format de tranche invalide : '{value}'. Attendu 'A - B' avec A < B.")
    low, high = int(match.group(1)), int(match.group(2))
    if low >= high and not (low == 0 and high == 0):
        raise ValueError(f"Borne inférieure ({low}) supérieure ou égale à borne supérieure ({high}).")
    return value


class HistoricalGameRow(BaseModel):
    """Ligne de données historique issue du catalogue Steam avec sentinelles documentées."""

    model_config = ConfigDict(extra="ignore")

    appid: int = Field(gt=0, description="Identifiant unique du jeu Steam")
    name: Optional[str] = Field(default=None, description="Titre du jeu")
    release_date: Optional[str] = Field(default=None, description="Date de sortie")
    required_age: Optional[int] = Field(default=0, description="Âge minimal (-1 est sentinelle historique)")
    price: Optional[float] = Field(default=0.0, ge=0.0, description="Prix d'achat standard")
    dlc_count: Optional[int] = Field(default=0, ge=0, description="Nombre de contenus téléchargeables")
    windows: bool = Field(default=True, description="Compatible Windows")
    mac: bool = Field(default=False, description="Compatible macOS")
    linux: bool = Field(default=False, description="Compatible Linux")
    achievements: Optional[int] = Field(default=0, ge=0, description="Nombre de succès déblocables")
    genres: Optional[str] = Field(default="[]", description="Genres sous forme textuelle ou liste JSON")
    positive: Optional[int] = Field(default=0, ge=0, description="Nombre d'avis positifs historiques")
    negative: Optional[int] = Field(default=0, ge=0, description="Nombre d'avis négatifs historiques")
    pct_pos_total: Optional[float] = Field(default=-1.0, description="Pourcentage d'avis positifs (-1 si < 10 avis)")
    num_reviews_total: Optional[int] = Field(default=0, ge=0, description="Nombre total d'avis utilisateurs")
    estimated_owners: Optional[str] = Field(default=None, description="Tranche de propriétaires estimés")

    @field_validator("estimated_owners")
    @classmethod
    def check_owner_range(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            return validate_owner_range_str(v)
        return v


class NewObservationGameRow(BaseModel):
    """Nouvelle observation labellisée (flux PULL ou PUSH). Rejette strictement les sentinelles."""

    model_config = ConfigDict(extra="forbid")

    appid: int = Field(gt=0, description="Identifiant Steam unique")
    name: Optional[str] = Field(default=None, description="Nom du jeu")
    release_date: Optional[str] = Field(default=None, description="Date de sortie valide")
    required_age: int = Field(ge=0, description="Âge requis >= 0 (sentinelle -1 interdite)")
    price: float = Field(ge=0.0, description="Prix >= 0.0")
    dlc_count: int = Field(ge=0, default=0, description="Nombre de DLCs")
    windows: bool = Field(default=True)
    mac: bool = Field(default=False)
    linux: bool = Field(default=False)
    achievements: int = Field(ge=0, default=0)
    genres: Union[str, List[str]] = Field(default="[]")
    pct_pos_total: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    num_reviews_total: Optional[int] = Field(default=None, ge=0)
    estimated_owners: Optional[str] = Field(default=None)

    @field_validator("estimated_owners")
    @classmethod
    def check_new_owner_range(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            return validate_owner_range_str(v)
        return v


# ==============================================================================
# 2. Contrats d'Inférence (API Prédiction - Anti-Leakage)
# ==============================================================================

class InferenceInputGame(BaseModel):
    """Entrée d'inférence unitaire.

    RÈGLE ABSOLUE ANTI-FUITE :
    Interdiction formelle des colonnes cibles et de fuite :
    - pct_pos_total, estimated_owners, num_reviews_total, positive, negative.
    Toute tentative de transmettre ces attributs déclenche une erreur 422 (extra='forbid').
    """

    model_config = ConfigDict(extra="forbid")

    appid: int = Field(gt=0, description="Identifiant du jeu pour audit et corrélation")
    name: Optional[str] = Field(default=None, description="Nom facultatif (exclu des features)")
    release_date: Optional[str] = Field(default=None, description="Date de sortie (ex: '2023-05-15')")
    price: Optional[float] = Field(default=0.0, ge=0.0, description="Prix du jeu en devise unitaire")
    required_age: Optional[int] = Field(default=0, ge=0, description="Âge requis")
    dlc_count: Optional[int] = Field(default=0, ge=0, description="Nombre de DLCs")
    achievements: Optional[int] = Field(default=0, ge=0, description="Nombre de succès")
    windows: bool = Field(default=True, description="Compatibilité Windows")
    mac: bool = Field(default=False, description="Compatibilité macOS")
    linux: bool = Field(default=False, description="Compatibilité Linux")
    genres: Union[str, List[str]] = Field(
        default="[]", description="Genres sous forme de liste ou chaîne textuelle"
    )


class InferenceBatchRequest(BaseModel):
    """Requête de prédiction par lot (1 à 100 jeux)."""

    model_config = ConfigDict(extra="forbid")

    items: List[InferenceInputGame] = Field(
        min_length=1, max_length=100, description="Liste ordonnée de jeux à prédire"
    )


class GamePredictionOutput(BaseModel):
    """Résultat prédictif pour un jeu donné couvrant les deux tâches."""

    appid: int
    predicted_pct: Optional[float] = Field(
        default=None, ge=0.0, le=100.0, description="Score prédit pour la régression"
    )
    predicted_owners: Optional[str] = Field(
        default=None, description="Tranche prédite pour la classification"
    )


class InferenceResponse(BaseModel):
    """Réponse standardisée de l'API d'inférence."""

    request_id: UUID = Field(description="Identifiant unique de la transaction HTTP")
    model_versions: Dict[str, str] = Field(
        description="Identifiants exacts des modèles mobilisés (régression & classification)"
    )
    results: List[GamePredictionOutput]
    inference_duration_ms: float = Field(ge=0.0, description="Durée de calcul prédictif")


# ==============================================================================
# 3. Contrat de Feedback Utilisateur
# ==============================================================================

class FeedbackInput(BaseModel):
    """Entrée de retour utilisateur/vérité terrain avec vérification d'idempotence."""

    model_config = ConfigDict(extra="forbid")

    feedback_id: UUID = Field(default_factory=uuid4, description="Identifiant unique fourni par le client")
    request_id: UUID = Field(description="Référence à la requête API d'origine")
    prediction_id: UUID = Field(description="Référence à la prédiction évaluée")
    task: Literal["regression", "classification"] = Field(description="Tâche concernée")
    useful: Optional[bool] = Field(default=None, description="Jugement qualitatif subjectif")
    observed_pct: Optional[float] = Field(
        default=None, ge=0.0, le=100.0, description="Vérité terrain observée pour régression"
    )
    observed_owners: Optional[str] = Field(
        default=None, description="Vérité terrain observée pour classification"
    )
    observed_at: Optional[datetime] = Field(default=None, description="Horodatage de l'observation")
    label_source: Optional[str] = Field(default=None, description="Origine de la vérité terrain")

    @model_validator(mode="after")
    def validate_feedback_constraints(self) -> "FeedbackInput":
        if self.useful is None and self.observed_pct is None and self.observed_owners is None:
            raise ValueError("Au moins l'un des champs useful, observed_pct ou observed_owners doit être fourni.")

        if self.task == "regression":
            if self.observed_owners is not None:
                raise ValueError("observed_owners doit être nul pour une tâche de régression.")
        elif self.task == "classification":
            if self.observed_pct is not None:
                raise ValueError("observed_pct doit être nul pour une tâche de classification.")
            if self.observed_owners is not None:
                validate_owner_range_str(self.observed_owners)

        # Cohérence de l'audit d'observation
        has_observed = self.observed_pct is not None or self.observed_owners is not None
        if has_observed and (self.observed_at is None or not self.label_source):
            # Pour la flexibilité client, si observed_at manque lors de l'envoi d'un label, on l'horodate
            if self.observed_at is None:
                self.observed_at = datetime.now(timezone.utc)
            if not self.label_source:
                self.label_source = "user_feedback"

        return self


# ==============================================================================
# 4. Structures Transmises inter-tâches (Prefect & Pipelines)
# ==============================================================================

class RunContext(BaseModel):
    """Contexte d'exécution transverse corrélant les identifiants techniques."""

    run_id: str
    mode: str = "training"
    batch_id: Optional[str] = None
    attempt: int = 1
    git_revision: str = "unknown"
    flow_run_id: Optional[str] = None
    mlflow_run_id: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class SnapshotRef(BaseModel):
    """Référence immuable vers un snapshot de données vérifié."""

    path: str
    sha256: str
    row_count: int
    column_count: int
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class IngestionResult(BaseModel):
    """Résultat produit par la tâche load_data."""

    context: RunContext
    snapshot: SnapshotRef
    gx_success: bool = True
    report_path: Optional[str] = None


class TaskDataset(BaseModel):
    """Partitions et caractéristiques d'une tâche d'apprentissage spécifique."""

    task_name: Literal["regression", "classification"]
    train_appids: List[int]
    val_appids: List[int]
    test_appids: List[int]
    feature_names: List[str]
    target_name: str


class PreparedData(BaseModel):
    """Résultat produit par la tâche preprocess_data."""

    context: RunContext
    regression_data: TaskDataset
    classification_data: TaskDataset
    transformers_info: Dict[str, Any] = Field(default_factory=dict)
