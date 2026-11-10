"""src/data_sources/cache.py - Cache local persistant avec contrôle de fraîcheur.

Permet :
- L'enregistrement horodaté des réponses JSON brutes des APIs Steam.
- Le respect d'une ancienneté maximale (ex. 24h pour SteamSpy).
- Le rejeu complet déconnecté des collectes sans sollicitation réseau.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Optional


class LocalAPICache:
    """Gestionnaire de cache sur disque pour les réponses d'APIs externes."""

    def __init__(self, cache_dir: Optional[Path | str] = None) -> None:
        if cache_dir:
            self.cache_dir = Path(cache_dir).resolve()
        else:
            self.cache_dir = (Path(__file__).parent.parent.parent / "data" / "cache").resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, source: str, appid: int) -> Path:
        """Génère le chemin canonique du fichier cache pour une source et un appid."""
        clean_source = source.strip().lower().replace("/", "_")
        return self.cache_dir / f"{clean_source}_{appid}.json"

    def get(
        self,
        source: str,
        appid: int,
        max_age_hours: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Récupère une réponse en cache si présente et non expirée."""
        cache_file = self._get_cache_path(source, appid)
        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                envelope = json.load(f)

            if not isinstance(envelope, dict) or "payload" not in envelope:
                return None

            # Vérification de l'expiration
            if max_age_hours is not None:
                fetched_at_str = envelope.get("fetched_at")
                if not fetched_at_str:
                    return None
                fetched_at = datetime.fromisoformat(fetched_at_str)
                age_seconds = (datetime.now(timezone.utc) - fetched_at).total_seconds()
                if age_seconds > max_age_hours * 3600:
                    return None

            return envelope.get("payload")
        except Exception:
            return None

    def set(
        self,
        source: str,
        appid: int,
        payload: Any,
        params: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Enregistre une réponse brute avec métadonnées horodatées."""
        cache_file = self._get_cache_path(source, appid)
        envelope = {
            "source": source,
            "appid": appid,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "params": params or {},
            "payload": payload,
        }
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(envelope, f, indent=2, ensure_ascii=False)
        return cache_file

    def has(self, source: str, appid: int, max_age_hours: Optional[float] = None) -> bool:
        """Vérifie la présence et validité d'une entrée en cache."""
        return self.get(source, appid, max_age_hours=max_age_hours) is not None
