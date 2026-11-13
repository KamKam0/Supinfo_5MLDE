"""api_prediction/schemas.py - Schémas Pydantic stricts d'inférence.

Garantit le respect absolu des contrats d'interface :
- Interdiction formelle de toute variable cible (extra='forbid').
- Support du mode unitaire (SinglePredictionRequest / InferenceGameInput) et par lot (BatchPredictionRequest).
- Réponses standardisées avec corrélation de requêtes, versions et durées.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, model_validator


class InferenceGameInput(BaseModel):
    """Entrée d'inférence pour un jeu vidéo Steam.

    RÈGLE ABSOLUE ANTI-FUITE :
    Les variables cibles et de fuite temporelle/métrique :
    `pct_pos_total`, `estimated_owners`, `num_reviews_total`, `positive`, `negative`
    sont formellement interdites et déclenchent une erreur HTTP 422 (extra='forbid').
    """

    model_config = ConfigDict(extra="forbid")

    appid: Optional[int] = Field(
        default=None, gt=0, description="Identifiant Steam unique pour audit et traçabilité"
    )
    name: Optional[str] = Field(
        default=None, description="Nom indicatif du jeu (non mobilisé par les estimateurs)"
    )
    price: Optional[float] = Field(
        default=0.0, ge=0.0, description="Prix du jeu en devise de référence (>= 0.0)"
    )
    required_age: Optional[int] = Field(
        default=0, ge=0, description="Âge légal minimum requis (>= 0)"
    )
    dlc_count: Optional[int] = Field(
        default=0, ge=0, description="Nombre de contenus téléchargeables (DLCs) (>= 0)"
    )
    achievements: Optional[int] = Field(
        default=0, ge=0, description="Nombre de succès déblocables (>= 0)"
    )
    windows: bool = Field(default=True, description="Compatibilité système Windows")
    mac: bool = Field(default=False, description="Compatibilité système macOS")
    linux: bool = Field(default=False, description="Compatibilité système Linux")
    genres: Union[str, List[str]] = Field(
        default_factory=list,
        description="Liste ou chaîne des genres déclarés (ex: ['Action', 'Indie'])",
    )
    release_date: Optional[str] = Field(
        default=None, description="Date de sortie (format recommandé YYYY-MM-DD)"
    )


class SinglePredictionRequest(BaseModel):
    """Conteneur d'entrée pour la prédiction unitaire sous clé 'game'."""

    model_config = ConfigDict(extra="forbid")

    game: InferenceGameInput = Field(
        ..., description="Caractéristiques du jeu soumis à l'inférence"
    )


class BatchPredictionRequest(BaseModel):
    """Conteneur d'entrée pour la prédiction vectorisée par lot (1 à 100 jeux)."""

    model_config = ConfigDict(extra="forbid")

    games: Optional[List[InferenceGameInput]] = Field(
        default=None,
        min_length=1,
        max_length=100,
        description="Liste ordonnée de jeux (1 à 100 éléments)",
    )
    items: Optional[List[InferenceGameInput]] = Field(
        default=None,
        min_length=1,
        max_length=100,
        description="Alias de 'games' (1 à 100 éléments)",
    )

    @model_validator(mode="after")
    def validate_payload(self) -> "BatchPredictionRequest":
        if self.games is None and self.items is None:
            raise ValueError(
                "Le champ 'games' (ou 'items') est obligatoire et doit contenir entre 1 et 100 jeux."
            )
        if self.games is not None and self.items is not None:
            raise ValueError("Fournir soit 'games', soit 'items', mais pas les deux simultanément.")
        return self

    def get_games(self) -> List[InferenceGameInput]:
        """Retourne la liste des jeux à traiter."""
        return self.games if self.games is not None else (self.items or [])


class TaskModelInfo(BaseModel):
    """Identité formelle du modèle ayant produit la prédiction."""

    server_uri: str = Field(description="URI du serveur MLflow de référence")
    model_name: str = Field(description="Nom du modèle dans le registre MLflow")
    model_version: str = Field(description="Numéro de version exact servi")
    mlflow_run_id: str = Field(description="Identifiant du run d'entraînement source")


class TaskPredictionResult(BaseModel):
    """Résultat prédictif unitaire pour une tâche spécifique."""

    prediction_id: str = Field(description="Identifiant unique de la prédiction")
    task: Literal["regression", "classification"] = Field(
        description="Type de tâche ML ('regression' ou 'classification')"
    )
    target_column: str = Field(
        description="Nom de la variable cible prédite ('pct_pos_total' ou 'estimated_owners')"
    )
    predicted_value: Any = Field(
        description="Valeur prédite (réel [0, 100] pour régression, tranche textuelle pour classification)"
    )
    unit: Optional[str] = Field(
        default=None, description="Unité de mesure ou label de référence"
    )
    model: TaskModelInfo = Field(
        description="Métadonnées d'audit du modèle mobilisé"
    )


class GamePredictionItem(BaseModel):
    """Résultat prédictif conjoint pour un jeu individuel."""

    item_index: int = Field(ge=0, description="Position séquentielle de l'élément (base 0)")
    appid: Optional[int] = Field(default=None, description="Identifiant Steam du jeu")
    predictions: List[TaskPredictionResult] = Field(
        description="Résultats prédictifs couvrant les deux tâches (régression & classification)"
    )
    predicted_pct: Optional[float] = Field(
        default=None, ge=0.0, le=100.0, description="Pourcentage d'avis positifs prédit [0, 100]"
    )
    predicted_owners: Optional[str] = Field(
        default=None, description="Tranche de propriétaires prédite"
    )


class PredictionResponse(BaseModel):
    """Structure commune standardisée de réponse d'inférence."""

    request_id: str = Field(description="Identifiant unique de la transaction HTTP de prédiction")
    manifest_id: str = Field(description="Identifiant du manifeste de service actif")
    item_count: int = Field(ge=1, description="Nombre total de jeux évalués dans cette transaction")
    results: List[GamePredictionItem] = Field(
        description="Liste ordonnée des résultats prédictifs"
    )
    processing_duration_ms: float = Field(
        ge=0.0, description="Durée totale de traitement côté API en millisecondes"
    )
    inference_duration_ms: float = Field(
        ge=0.0, description="Durée pure d'inférence des pipelines ML en millisecondes"
    )


class HealthResponse(BaseModel):
    """Réponse des sondes d'observabilité live et ready."""

    status: Literal["live", "ready", "not_ready"] = Field(description="État de santé")
    service: str = Field(default="steam-prediction-api", description="Nom du composant")
    instance_id: Optional[str] = Field(
        default=None, description="Identifiant unique du worker actif"
    )
    manifest_id: Optional[str] = Field(
        default=None, description="Identifiant du manifeste servi"
    )
    models_loaded: bool = Field(default=False, description="Indicateur de chargement des modèles")
    timestamp: str = Field(description="Horodatage UTC ISO 8601")
    error: Optional[str] = Field(default=None, description="Détail de l'incident éventuel")


class ModelsInfoResponse(BaseModel):
    """Détail du manifeste de service et des modèles actifs."""

    manifest_id: str = Field(description="Identifiant unique du manifeste")
    instance_id: str = Field(description="Identifiant du worker courant")
    mode: str = Field(description="Mode d'exploitation ('qualified' ou 'demo')")
    generated_at: str = Field(description="Date de génération du manifeste")
    manifest_sha256: str = Field(description="Empreinte cryptographique SHA-256 du manifeste")
    mlflow_tracking_uri: str = Field(description="URI MLflow de référence")
    models: Dict[str, Any] = Field(description="Dictionnaire des versions de modèles figées")


class ErrorResponse(BaseModel):
    """Enveloppe d'erreur standardisée sans divulgation d'informations sensibles."""

    request_id: str = Field(description="Identifiant de corrélation de la requête en échec")
    error_code: str = Field(
        description="Code d'erreur fonctionnel standard (INVALID_INPUT, DEPENDENCY_UNAVAILABLE, etc.)"
    )
    message: str = Field(description="Message explicatif clair et non confidentiel")
    details: Optional[Any] = Field(
        default=None, description="Détails techniques optionnels d'anomalie ou de validation"
    )


# ==============================================================================
# Contrats de Retours / Feedbacks
# ==============================================================================

class FeedbackInput(BaseModel):
    """Contrat d'entrée pour la réception de retours d'utilisation et vérités terrain."""

    model_config = ConfigDict(extra="forbid")

    feedback_id: str = Field(
        description="Identifiant UUID unique du retour client (utilisé pour détection d'idempotence)"
    )
    prediction_id: str = Field(
        description="Identifiant UUID de la prédiction unitaire ciblée"
    )
    useful: Optional[bool] = Field(
        default=None, description="Avis subjectif sur l'utilité de la prédiction"
    )
    observed_pct: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Pourcentage d'avis positifs réellement constaté (régression)",
    )
    observed_owners: Optional[str] = Field(
        default=None,
        description="Tranche de propriétaires réellement constatée (classification, ex: '20000 - 50000')",
    )
    observed_at: Optional[str] = Field(
        default=None,
        description="Date UTC d'observation de la vérité terrain (ISO 8601, obligatoire si observation)",
    )
    label_source: Optional[str] = Field(
        default=None,
        description="Source de la vérité terrain (ex: 'steam-store-actual', 'loadtest', etc.)",
    )

    @model_validator(mode="after")
    def validate_feedback_constraints(self) -> "FeedbackInput":
        has_opinion = self.useful is not None
        has_reg_obs = self.observed_pct is not None
        has_cls_obs = self.observed_owners is not None

        if not (has_opinion or has_reg_obs or has_cls_obs):
            raise ValueError(
                "Le feedback doit comporter au moins un avis subjectif ('useful') "
                "ou une observation réelle ('observed_pct' ou 'observed_owners')."
            )

        if has_reg_obs and has_cls_obs:
            raise ValueError(
                "Une observation ne peut porter simultanément sur la régression et la classification."
            )

        has_observation = has_reg_obs or has_cls_obs
        if has_observation:
            if not self.observed_at:
                raise ValueError("Toute observation réelle exige obligatoirement le champ 'observed_at'.")
            if not self.label_source or not self.label_source.strip():
                raise ValueError("Toute observation réelle exige obligatoirement le champ 'label_source'.")
        else:
            if self.observed_at is not None or self.label_source is not None:
                raise ValueError(
                    "Les métadonnées 'observed_at' et 'label_source' ne sont autorisées "
                    "qu'en présence d'une observation ('observed_pct' ou 'observed_owners')."
                )

        if has_cls_obs:
            import re
            m = re.match(r"^(\d{1,20})\s*-\s*(\d{1,20})$", str(self.observed_owners).strip())
            if not m:
                raise ValueError(
                    f"Format de tranche 'observed_owners' invalide : '{self.observed_owners}'. Format attendu: 'lower - upper'."
                )
            low, high = int(m.group(1)), int(m.group(2))
            if low >= high:
                raise ValueError(
                    f"Tranche invalide : borne inférieure ({low}) >= borne supérieure ({high})."
                )

        return self


class FeedbackResponse(BaseModel):
    """Réponse d'enregistrement ou de rejeu d'un feedback."""

    request_id: str = Field(description="Identifiant de transaction HTTP")
    feedback_id: str = Field(description="Identifiant unique du feedback")
    prediction_id: str = Field(description="Prédiction associée")
    status: Literal["created", "replayed"] = Field(
        description="Statut fonctionnel de l'opération ('created' pour HTTP 201, 'replayed' pour HTTP 200)"
    )
    message: str = Field(description="Message de confirmation")

