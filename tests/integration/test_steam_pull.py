"""tests/integration/test_steam_pull.py - Tests d'intégration de l'acquisition active PULL.

Vérifie :
1. Le respect du rate limiting (espacement >= 1,1s entre requêtes).
2. Le fonctionnement du rejeu hors-ligne à partir du cache local.
3. La normalisation conforme au schéma 16 colonnes.
4. L'immuabilité absolue du snapshot de base lors de l'upsert.
5. Le rejet propre lors du dépassement du budget d'attente (Retry-After > 60s).
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from unittest.mock import MagicMock, patch
import httpx
import pandas as pd
import pytest

from src.contracts import REQUIRED_SNAPSHOT_COLUMNS
from src.data_sources.cache import LocalAPICache
from src.data_sources.normalization import normalize_owner_range, normalize_steam_game, parse_release_date_iso
from src.data_sources.snapshots import compute_file_sha256, upsert_and_publish_snapshot
from src.data_sources.steam_client import SteamPULLClient
from src.exceptions import DataValidationError


# ==============================================================================
# 1. Tests de Normalisation
# ==============================================================================

def test_normalize_owner_range():
    """Vérifie la transformation des tranches SteamSpy vers le format canonique."""
    assert normalize_owner_range("10,000,000 .. 20,000,000") == "10000000 - 20000000"
    assert normalize_owner_range("0 .. 20,000") == "0 - 20000"
    assert normalize_owner_range("20000 - 50000") == "20000 - 50000"
    with pytest.raises(DataValidationError):
        normalize_owner_range("50000 - 20000")  # Bornes inversées


def test_parse_release_date_iso():
    """Vérifie le parsing multi-format de dates vers YYYY-MM-DD."""
    assert parse_release_date_iso("18 Apr, 2011") == "2011-04-18"
    assert parse_release_date_iso("2020-12-10") == "2020-12-10"
    assert parse_release_date_iso("10 Oct, 2007") == "2007-10-10"


def test_normalize_steam_game_nominal():
    """Vérifie l'assemblage complet des 16 colonnes depuis les 3 sources."""
    store_payload = {
        "620": {
            "success": True,
            "data": {
                "name": "Portal 2",
                "required_age": 0,
                "is_free": False,
                "price_overview": {"final": 999},
                "dlc": [123, 456],
                "platforms": {"windows": True, "mac": True, "linux": True},
                "achievements": {"total": 51},
                "genres": [{"description": "Action"}, {"description": "Adventure"}],
                "release_date": {"coming_soon": False, "date": "18 Apr, 2011"},
            },
        }
    }
    steamspy_payload = {
        "appid": 620,
        "owners": "10,000,000 .. 20,000,000",
        "positive": 380000,
        "negative": 5000,
    }
    game = normalize_steam_game(620, store_payload, steamspy_payload)

    assert game["appid"] == 620
    assert game["name"] == "Portal 2"
    assert game["release_date"] == "2011-04-18"
    assert game["price"] == 9.99
    assert game["dlc_count"] == 2
    assert game["achievements"] == 51
    assert game["windows"] is True
    assert game["mac"] is True
    assert game["linux"] is True
    assert game["num_reviews_total"] == 385000
    assert game["pct_pos_total"] == round(380000 / 385000 * 100, 2)
    assert game["estimated_owners"] == "10000000 - 20000000"


def test_normalize_steam_game_rejects_unreleased():
    """Un jeu non publié (coming_soon=True) doit être rejeté."""
    store_payload = {
        "999": {
            "success": True,
            "data": {
                "name": "Unreleased Game",
                "release_date": {"coming_soon": True, "date": "2027"},
            },
        }
    }
    with pytest.raises(DataValidationError):
        normalize_steam_game(999, store_payload)


# ==============================================================================
# 2. Tests de Rate Limiting & Budget Réseau
# ==============================================================================

def test_rate_limiting_enforced(tmp_path: Path):
    """Vérifie que l'intervalle entre requêtes consécutives est au moins de 1,1 seconde."""
    cache = LocalAPICache(cache_dir=tmp_path / "cache")
    client = SteamPULLClient(cache=cache, timeout=5.0)

    t0 = time.time()
    client._throttle()
    t1 = time.time()
    client._throttle()
    t2 = time.time()

    elapsed = t2 - t1
    assert elapsed >= 1.05  # Marge de précision tolérée (1.1s visé)


def test_retry_after_over_limit_aborts(tmp_path: Path):
    """Une attente Retry-After supérieure à 60s doit interrompre la collecte immédiatement."""
    cache = LocalAPICache(cache_dir=tmp_path / "cache")
    client = SteamPULLClient(cache=cache, timeout=5.0)

    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.headers = {"Retry-After": "120"}

    with patch("httpx.Client.get", return_value=mock_resp):
        with pytest.raises(DataValidationError) as exc_info:
            client._request_with_retry("https://api.test.com")
        assert "60.0s" in str(exc_info.value)


# ==============================================================================
# 3. Cache Local & Rejeu Hors-Ligne
# ==============================================================================

def test_offline_replay_from_cache(tmp_path: Path):
    """Vérifie que le rejeu hors-ligne fonctionne à 100% à partir du cache sans appel réseau."""
    cache_dir = tmp_path / "cache"
    cache = LocalAPICache(cache_dir=cache_dir)

    store_payload = {
        "620": {
            "success": True,
            "data": {
                "name": "Portal 2",
                "required_age": 0,
                "is_free": False,
                "price_overview": {"final": 999},
                "platforms": {"windows": True, "mac": True, "linux": True},
                "achievements": {"total": 51},
                "genres": [{"description": "Action"}],
                "release_date": {"coming_soon": False, "date": "18 Apr, 2011"},
            },
        }
    }
    spy_payload = {
        "appid": 620,
        "owners": "10,000,000 .. 20,000,000",
        "positive": 380000,
        "negative": 5000,
    }

    cache.set("appdetails", 620, store_payload)
    cache.set("steamspy", 620, spy_payload)

    # Client configuré en mode déconnecté (offline_mode=True)
    offline_client = SteamPULLClient(cache=cache, offline_mode=True)
    game = offline_client.fetch_and_normalize_game(620)

    assert game["appid"] == 620
    assert game["name"] == "Portal 2"
    assert game["price"] == 9.99
    assert game["estimated_owners"] == "10000000 - 20000000"


# ==============================================================================
# 4. Upsert & Immuabilité du Snapshot de Base
# ==============================================================================

def test_upsert_immutability_and_row_counts(tmp_path: Path):
    """Vérifie que l'upsert préserve bit-à-bit le snapshot de base et incrémente le compte."""
    # Création d'un mini-snapshot de base conforme
    base_rows = []
    for i in range(1, 11):
        base_rows.append(
            {
                "appid": i,
                "name": f"Base Game {i}",
                "release_date": f"2020-{i:02d}-15",
                "required_age": 0,
                "price": 9.99,
                "dlc_count": 0,
                "achievements": 10,
                "windows": True,
                "mac": False,
                "linux": False,
                "positive": 100,
                "negative": 10,
                "num_reviews_total": 110,
                "pct_pos_total": 90.91,
                "genres": "['Action']",
                "estimated_owners": "20000 - 50000",
            }
        )
    base_df = pd.DataFrame(base_rows)[list(REQUIRED_SNAPSHOT_COLUMNS)]
    base_file = tmp_path / "base_snapshot.csv"
    base_df.to_csv(base_file, index=False, lineterminator="\r\n")
    initial_sha = compute_file_sha256(base_file)

    # Nouvelles observations : 1 mise à jour (appid 1) + 2 ajouts (appids 11, 12)
    new_obs = [
        {
            "appid": 1,
            "name": "Base Game 1 Updated",
            "release_date": "2020-01-15",
            "required_age": 0,
            "price": 14.99,
            "dlc_count": 1,
            "achievements": 15,
            "windows": True,
            "mac": False,
            "linux": False,
            "positive": 150,
            "negative": 10,
            "num_reviews_total": 160,
            "pct_pos_total": 93.75,
            "genres": "['Action', 'RPG']",
            "estimated_owners": "50000 - 100000",
        },
        {
            "appid": 11,
            "name": "New Game 11",
            "release_date": "2022-05-20",
            "required_age": 0,
            "price": 19.99,
            "dlc_count": 0,
            "achievements": 20,
            "windows": True,
            "mac": True,
            "linux": False,
            "positive": 200,
            "negative": 20,
            "num_reviews_total": 220,
            "pct_pos_total": 90.91,
            "genres": "['Indie']",
            "estimated_owners": "20000 - 50000",
        },
        {
            "appid": 12,
            "name": "New Game 12",
            "release_date": "2023-08-11",
            "required_age": 18,
            "price": 29.99,
            "dlc_count": 2,
            "achievements": 30,
            "windows": True,
            "mac": False,
            "linux": True,
            "positive": 500,
            "negative": 50,
            "num_reviews_total": 550,
            "pct_pos_total": 90.91,
            "genres": "['Strategy']",
            "estimated_owners": "100000 - 200000",
        },
    ]

    out_file = tmp_path / "output_snapshot.csv"
    summary = upsert_and_publish_snapshot(
        base_snapshot_path=base_file,
        new_observations=new_obs,
        output_snapshot_path=out_file,
        validate_before_merge=True,
        validate_after_merge=False,  # Pas de seuil 1000 sur le mini-jeu de test
    )

    # 1. Vérification d'immuabilité
    assert compute_file_sha256(base_file) == initial_sha

    # 2. Vérification des comptes
    assert summary["base_row_count"] == 10
    assert summary["updated_count"] == 1
    assert summary["added_count"] == 2
    assert summary["final_row_count"] == 12

    # 3. Vérification du contenu consolidé
    result_df = pd.read_csv(out_file)
    assert len(result_df) == 12
    game1 = result_df[result_df["appid"] == 1].iloc[0]
    assert game1["name"] == "Base Game 1 Updated"
    assert float(game1["price"]) == 14.99
