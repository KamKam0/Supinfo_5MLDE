"""src/data_sources/steam_client.py - Client HTTP Steam avec budget réseau, rate limiting et cache.

Garantit :
- Un espacement obligatoire d'au moins 1,1s entre requêtes réseau consécutives.
- Un timeout strict de 20s par requête.
- Jusqu'à 3 tentatives sur erreurs temporaires (429, 503) avec respect de Retry-After.
- Interruption immédiate si l'attente requise dépasse 60s.
- Utilisation prioritaire du cache local pour le travail hors-ligne et les tests déterministes.
"""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import httpx

from src.data_sources.cache import LocalAPICache
from src.data_sources.normalization import normalize_steam_game
from src.exceptions import DataValidationError

logger = logging.getLogger(__name__)


class SteamPULLClient:
    """Client d'acquisition PULL respectant les limites de budget réseau et le cache local."""

    MIN_REQUEST_INTERVAL: float = 1.1  # Secondes minimales entre requêtes externes
    DEFAULT_TIMEOUT: float = 20.0      # Secondes de timeout par requête
    MAX_RETRIES: int = 3               # Nombre maximal de tentatives sur erreur temporaire
    MAX_WAIT_ALLOWED: float = 60.0     # Attente maximale tolérée (Retry-After) avant abandon

    STORE_URL = "https://store.steampowered.com/api/appdetails"
    REVIEWS_URL = "https://store.steampowered.com/appreviews/{appid}"
    STEAMSPY_URL = "https://steamspy.com/api.php"

    def __init__(
        self,
        cache: Optional[LocalAPICache] = None,
        timeout: float = DEFAULT_TIMEOUT,
        offline_mode: bool = False,
    ) -> None:
        self.cache = cache or LocalAPICache()
        self.timeout = timeout
        self.offline_mode = offline_mode
        self._last_request_time: float = 0.0

    def _throttle(self) -> None:
        """Applique l'espacement obligatoire d'au moins 1,1 seconde entre requêtes consécutives."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.MIN_REQUEST_INTERVAL:
            sleep_duration = self.MIN_REQUEST_INTERVAL - elapsed
            time.sleep(sleep_duration)
        self._last_request_time = time.time()

    def _request_with_retry(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Exécute une requête HTTP avec gestion des réessais bornés et respect de Retry-After."""
        if self.offline_mode:
            return None

        req_headers = {
            "User-Agent": "SteamMLOpsProject/1.0 (Course 5MLDE; Python httpx client)",
            "Accept": "application/json",
        }
        if headers:
            req_headers.update(headers)

        attempts = 0
        while attempts < self.MAX_RETRIES:
            attempts += 1
            self._throttle()

            try:
                with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
                    resp = client.get(url, params=params, headers=req_headers)

                # Cas 1 : Succès
                if resp.status_code == 200:
                    try:
                        return resp.json()
                    except Exception:
                        return None

                # Cas 2 : Erreur temporaire de débit (429) ou indisponibilité (503)
                if resp.status_code in (429, 503):
                    retry_after_raw = resp.headers.get("Retry-After")
                    wait_seconds = 2.0 * attempts
                    if retry_after_raw:
                        try:
                            wait_seconds = float(retry_after_raw)
                        except ValueError:
                            pass

                    if wait_seconds > self.MAX_WAIT_ALLOWED:
                        raise DataValidationError(
                            f"Collecte interrompue : l'API a demandé une attente de {wait_seconds:.1f}s, "
                            f"dépassant le seuil maximal toléré de {self.MAX_WAIT_ALLOWED}s."
                        )

                    logger.warning(
                        f"Code HTTP {resp.status_code} sur {url}. Attente de {wait_seconds:.1f}s "
                        f"(tentative {attempts}/{self.MAX_RETRIES})..."
                    )
                    time.sleep(wait_seconds)
                    continue

                # Cas 3 : Erreurs permanentes (404, 400, etc.) -> pas de retry aveugle
                logger.warning(f"Réponse HTTP permanente {resp.status_code} sur {url}.")
                return None

            except httpx.RequestError as exc:
                if attempts >= self.MAX_RETRIES:
                    logger.error(f"Échec réseau définitif après {attempts} tentatives : {exc}")
                    return None
                time.sleep(1.5 * attempts)

        return None

    def fetch_store_details(self, appid: int, force_refresh: bool = False) -> Dict[str, Any]:
        """Récupère les métadonnées de la boutique Steam (depuis le cache ou l'API)."""
        source_key = "appdetails"
        if not force_refresh:
            cached = self.cache.get(source_key, appid)
            if cached is not None:
                return cached

        params = {"appids": str(appid), "cc": "us", "l": "english"}
        data = self._request_with_retry(self.STORE_URL, params=params)

        if data is not None and str(appid) in data and data[str(appid)].get("success"):
            self.cache.set(source_key, appid, data, params=params)
            return data

        # Repli sur le cache même expiré si disponible
        fallback = self.cache.get(source_key, appid)
        if fallback is not None:
            return fallback

        raise DataValidationError(f"Impossible de récupérer les métadonnées store pour l'appid {appid}.")

    def fetch_steamspy_details(self, appid: int, force_refresh: bool = False) -> Dict[str, Any]:
        """Récupère les statistiques SteamSpy (fraîcheur minimale de 24h respectée)."""
        source_key = "steamspy"
        if not force_refresh:
            # Fraîcheur de 24 heures pour SteamSpy
            cached = self.cache.get(source_key, appid, max_age_hours=24.0)
            if cached is not None:
                return cached

        params = {"request": "appdetails", "appid": str(appid)}
        data = self._request_with_retry(self.STEAMSPY_URL, params=params)

        if data is not None and isinstance(data, dict) and data.get("appid"):
            self.cache.set(source_key, appid, data, params=params)
            return data

        fallback = self.cache.get(source_key, appid)
        if fallback is not None:
            return fallback

        raise DataValidationError(f"Impossible de récupérer les statistiques SteamSpy pour l'appid {appid}.")

    def fetch_reviews_summary(self, appid: int, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """Récupère le résumé des avis Steam (avec repli gracieux)."""
        source_key = "appreviews"
        if not force_refresh:
            cached = self.cache.get(source_key, appid)
            if cached is not None:
                return cached

        url = self.REVIEWS_URL.format(appid=appid)
        params = {"json": "1", "language": "all", "purchase_type": "all"}
        data = self._request_with_retry(url, params=params)

        if data is not None and isinstance(data, dict) and "query_summary" in data:
            self.cache.set(source_key, appid, data, params=params)
            return data

        return self.cache.get(source_key, appid)

    def fetch_and_normalize_game(self, appid: int, force_refresh: bool = False) -> Dict[str, Any]:
        """Acquiert l'ensemble des sources et produit la ligne canonique 16 colonnes."""
        store_payload = self.fetch_store_details(appid, force_refresh=force_refresh)
        steamspy_payload = self.fetch_steamspy_details(appid, force_refresh=force_refresh)
        reviews_payload = self.fetch_reviews_summary(appid, force_refresh=force_refresh)

        return normalize_steam_game(
            appid=appid,
            store_payload=store_payload,
            steamspy_payload=steamspy_payload,
            reviews_payload=reviews_payload,
        )

    def fetch_batch(self, appids: List[int], force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Acquiert et normalise séquentiellement un lot d'appids avec rate limiting garanti."""
        results: List[Dict[str, Any]] = []
        for appid in appids:
            try:
                game = self.fetch_and_normalize_game(appid, force_refresh=force_refresh)
                results.append(game)
                print(f"[PULL] AppID {appid} ({game['name']}) acquis et normalisé avec succès.")
            except Exception as exc:
                print(f"[PULL] Erreur d'acquisition sur AppID {appid} : {exc}")
                raise exc
        return results
