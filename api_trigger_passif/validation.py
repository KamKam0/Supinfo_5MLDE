"""api_trigger_passif/validation.py - Contrats Pydantic pour l'API de collecte passive.

Définit le schéma d'un événement PUSH labellisé et ses règles de validation strictes :
- event_id, source et observed_at obligatoires.
- Objet game complet avec les deux cibles (pct_pos_total et estimated_owners).
- Interdiction formelle des sentinelles et des tranches '0 - 0'.
"""

from __future__ import annotations

from datetime import datetime
import re
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator


OWNER_RANGE_REGEX = re.compile(r"^(\d{1,20})\s*-\s*(\d{1,20})$")


def validate_push_owner_range(value: str) -> str:
    """Valide qu'une tranche d'owners PUSH est valide, avec bornes strictement croissantes (exclut '0 - 0')."""
    s = str(value).strip().replace(",", "")
    match = OWNER_RANGE_REGEX.match(s)
    if not match:
        raise ValueError(f"Format de tranche d'owners invalide : '{value}'. Attendu 'A - B'.")
    low, high = int(match.group(1)), int(match.group(2))
    if low >= high:
        raise ValueError(f"Bornes de possession invalides ou égales : '{value}' ({low} >= {high}).")
    if low == 0 and high == 0:
        raise ValueError("La tranche '0 - 0' est formellement interdite pour les nouvelles observations.")
    return f"{low} - {high}"


class PushGameData(BaseModel):
    """Données complètes d'un jeu labellisé reçu par le service PUSH."""

    model_config = ConfigDict(extra="ignore")

    appid: int = Field(gt=0, description="Identifiant unique du jeu (entier positif 64 bits)")
    name: str = Field(min_length=1, description="Nom officiel du jeu")
    release_date: Optional[str] = Field(default="2024-01-01", description="Date de sortie ISO YYYY-MM-DD")
    required_age: int = Field(default=0, ge=0, description="Âge requis >= 0 (pas de sentinelle -1)")
    price: float = Field(default=0.0, ge=0.0, description="Prix d'achat standard >= 0.0")
    dlc_count: int = Field(default=0, ge=0, description="Nombre de DLCs")
    windows: bool = Field(default=True, description="Compatible Windows")
    mac: bool = Field(default=False, description="Compatible macOS")
    linux: bool = Field(default=False, description="Compatible Linux")
    achievements: int = Field(default=0, ge=0, description="Nombre de succès")
    genres: Union[str, List[str]] = Field(default="['Action']", description="Genres")
    positive: int = Field(default=100, ge=0, description="Avis positifs")
    negative: int = Field(default=10, ge=0, description="Avis négatifs")
    num_reviews_total: int = Field(ge=0, description="Nombre total d'avis")
    pct_pos_total: float = Field(ge=0.0, le=100.0, description="Pourcentage d'avis positifs dans [0, 100]")
    estimated_owners: str = Field(description="Tranche de propriétaires (A < B, non nul)")

    @field_validator("estimated_owners")
    @classmethod
    def check_owners(cls, v: str) -> str:
        return validate_push_owner_range(v)

    @field_validator("genres")
    @classmethod
    def format_genres_str(cls, v: Union[str, List[str]]) -> str:
        if isinstance(v, list):
            return str([str(x).strip() for x in v if str(x).strip()])
        return str(v)


class PushEventPayload(BaseModel):
    """Payload d'un événement HTTP soumis à POST /games."""

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1, max_length=128, description="Identifiant unique de l'événement")
    source: str = Field(default="simulator", max_length=64, description="Source d'émission")
    observed_at: datetime = Field(description="Horodatage d'observation avec fuseau horaire")
    game: PushGameData = Field(description="Observation complète labellisée")
