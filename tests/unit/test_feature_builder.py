"""tests/unit/test_feature_builder.py - Tests unitaires du transformeur SteamFeatureBuilder."""

import numpy as np
import pandas as pd
import pytest

from src.feature_builder import SteamFeatureBuilder, safe_parse_genres


def test_safe_parse_genres():
    assert safe_parse_genres("['Action', 'RPG']") == ["action", "rpg"]
    assert safe_parse_genres("Action, Adventure") == ["action", "adventure"]
    assert safe_parse_genres("[]") == []
    assert safe_parse_genres(None) == []
    assert safe_parse_genres(np.nan) == []
    assert safe_parse_genres(["Strategy", "Strategy", "Indie"]) == ["strategy", "indie"]


def test_steam_feature_builder_fit_transform():
    df = pd.DataFrame([
        {
            "price": 19.99,
            "required_age": 12,
            "dlc_count": 2,
            "achievements": 20,
            "windows": True,
            "mac": False,
            "linux": False,
            "genres": "['Action', 'Indie']",
            "release_date": "2020-05-15",
        },
        {
            "price": 0.0,
            "required_age": -1,  # Sentinelle
            "dlc_count": 0,
            "achievements": 0,
            "windows": True,
            "mac": True,
            "linux": True,
            "genres": "['Free To Play', 'Strategy']",
            "release_date": "2018-11-20",
        },
        {
            "price": None,  # Manquant
            "required_age": 0,
            "dlc_count": None,
            "achievements": None,
            "windows": False,
            "mac": False,
            "linux": False,
            "genres": "[]",
            "release_date": None,
        },
    ])

    fb = SteamFeatureBuilder(top_genres_k=5)
    matrix = fb.fit_transform(df)

    assert isinstance(matrix, np.ndarray)
    assert matrix.shape[0] == 3
    assert matrix.shape[1] == len(fb.get_feature_names_out())
    # Aucune valeur NaN ou infinie dans la matrice finale
    assert not np.isnan(matrix).any()
    assert not np.isinf(matrix).any()


def test_steam_feature_builder_handles_unseen_data():
    """Vérifie que des genres inconnus et des valeurs aberrantes à l'inférence ne provoquent pas d'erreur."""
    train_df = pd.DataFrame([
        {"price": 10.0, "genres": "['Action']", "release_date": "2021-01-01"},
        {"price": 20.0, "genres": "['Adventure']", "release_date": "2021-06-01"},
    ])
    test_df = pd.DataFrame([
        {"price": 15.0, "genres": "['CompletelyNewGenre_123']", "release_date": "InvalidDate"},
    ])

    fb = SteamFeatureBuilder()
    fb.fit(train_df)
    transformed = fb.transform(test_df)

    assert transformed.shape[0] == 1
    assert not np.isnan(transformed).any()
