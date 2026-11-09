"""src/feature_builder.py - Transformeur scikit-learn de caractéristiques Steam.

Transformeur déterministe pour le prétraitement des métadonnées du catalogue Steam :
- Parsing sécurisé des genres textuels et extraction des indicateurs booléens.
- Transformations logarithmiques log1p sur le prix, les DLCs et les succès.
- Encodage cyclique de la saisonnalité (sin/cos du mois) et extraction de l'année.
- Gestion des sentinelles (-1 dans required_age -> NaN).
- Apprentissage des paramètres (médianes, vocabulaire des genres) strictement lors du fit() sur Train.
- Traitement sécurisé des valeurs inconnues à l'inférence.
"""

from __future__ import annotations

import ast
import re
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Liste des genres clés à extraire sous forme d'indicateurs binaires
CORE_GENRE_FLAGS = [
    "indie",
    "action",
    "adventure",
    "casual",
    "simulation",
    "strategy",
    "rpg",
    "early access",
    "free to play",
    "massively multiplayer",
    "racing",
    "sports",
]


def safe_parse_genres(val: Any) -> List[str]:
    """Parse de manière sécurisée (sans jamais appeler eval()) une liste de genres."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []

    raw_list: List[str] = []
    if isinstance(val, (list, tuple)):
        raw_list = [str(x).strip().lower() for x in val if str(x).strip()]
    else:
        val_str = str(val).strip()
        if not val_str or val_str in ("[]", "set()", "nan", "none"):
            return []
        try:
            parsed = ast.literal_eval(val_str)
            if isinstance(parsed, (list, tuple)):
                raw_list = [str(x).strip().lower() for x in parsed if str(x).strip()]
        except Exception:
            pass

        if not raw_list:
            cleaned = re.sub(r"[\[\]'\"()]", "", val_str)
            raw_list = [p.strip().lower() for p in re.split(r"[,;]", cleaned) if p.strip()]

    # Dédoublonnage en préservant l'ordre
    seen = set()
    result = []
    for p in raw_list:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result


class SteamFeatureBuilder(BaseEstimator, TransformerMixin):
    """Transformeur de caractéristiques scikit-learn pour le catalogue Steam."""

    def __init__(self, top_genres_k: int = 15, default_year: float = 2020.0):
        self.top_genres_k = top_genres_k
        self.default_year = default_year
        self.fitted_ = False
        self.learned_medians_: Dict[str, float] = {}
        self.top_genres_: List[str] = []
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> "SteamFeatureBuilder":
        """Apprend les statistiques de centrage et le vocabulaire des genres sur Train uniquement."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # 1. Vocabulaire des genres principaux
        primary_genres: List[str] = []
        for val in X.get("genres", []):
            parsed = safe_parse_genres(val)
            primary_genres.append(parsed[0] if parsed else "unknown")

        series_genres = pd.Series(primary_genres)
        val_counts = series_genres.value_counts()
        # Conserver les top K genres (hors unknown)
        known_counts = val_counts[val_counts.index != "unknown"]
        self.top_genres_ = list(known_counts.head(self.top_genres_k).index)

        # 2. Médianes des colonnes numériques
        # Prix
        prices = pd.to_numeric(X.get("price", pd.Series()), errors="coerce")
        prices_clean = prices[prices >= 0.0]
        self.learned_medians_["price"] = float(prices_clean.median()) if not prices_clean.empty else 4.99

        # required_age (sentinelle -1 convertie en NaN)
        ages = pd.to_numeric(X.get("required_age", pd.Series()), errors="coerce")
        ages_clean = ages[ages >= 0]
        self.learned_medians_["required_age"] = float(ages_clean.median()) if not ages_clean.empty else 0.0

        # DLC count
        dlcs = pd.to_numeric(X.get("dlc_count", pd.Series()), errors="coerce")
        dlcs_clean = dlcs[dlcs >= 0]
        self.learned_medians_["dlc_count"] = float(dlcs_clean.median()) if not dlcs_clean.empty else 0.0

        # Achievements
        achs = pd.to_numeric(X.get("achievements", pd.Series()), errors="coerce")
        achs_clean = achs[achs >= 0]
        self.learned_medians_["achievements"] = float(achs_clean.median()) if not achs_clean.empty else 0.0

        # Année de sortie
        years: List[float] = []
        for d in X.get("release_date", []):
            try:
                dt = pd.to_datetime(d, errors="coerce")
                if pd.notna(dt) and 1990 <= dt.year <= 2030:
                    years.append(float(dt.year))
            except Exception:
                pass
        self.learned_medians_["release_year"] = (
            float(np.median(years)) if years else self.default_year
        )

        # Construction de la liste figée des noms de features
        self.feature_names_ = [
            "log1p_price",
            "is_free",
            "required_age",
            "log1p_dlc_count",
            "log1p_achievements",
            "windows",
            "mac",
            "linux",
            "platform_count",
            "release_year",
            "sin_month",
            "cos_month",
            "genre_count",
        ]
        # Indicateurs de genres clés
        for flag in CORE_GENRE_FLAGS:
            self.feature_names_.append(f"genre_flag_{flag.replace(' ', '_')}")
        # One-hot des top genres principaux
        for g in self.top_genres_:
            self.feature_names_.append(f"primary_genre_{g.replace(' ', '_')}")
        self.feature_names_.append("primary_genre_other")

        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Transforme les données brutes en matrice de caractéristiques numériques."""
        if not self.fitted_:
            raise RuntimeError("Le transformeur SteamFeatureBuilder doit être ajusté avec fit() avant transform().")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        n_samples = len(X)
        if n_samples == 0:
            return np.empty((0, len(self.feature_names_)))

        # 1. Traitement Prix
        raw_prices = pd.to_numeric(X.get("price", pd.Series(index=X.index)), errors="coerce")
        clean_prices = raw_prices.fillna(self.learned_medians_["price"]).clip(lower=0.0).values
        log1p_price = np.log1p(clean_prices)
        is_free = (clean_prices == 0.0).astype(float)

        # 2. Âge requis (-1 ou NaN remplacé par médiane)
        raw_ages = pd.to_numeric(X.get("required_age", pd.Series(index=X.index)), errors="coerce")
        raw_ages = raw_ages.mask(raw_ages < 0, np.nan)
        clean_ages = raw_ages.fillna(self.learned_medians_["required_age"]).clip(lower=0.0).values

        # 3. DLC et Achievements
        raw_dlc = pd.to_numeric(X.get("dlc_count", pd.Series(index=X.index)), errors="coerce")
        clean_dlc = raw_dlc.fillna(self.learned_medians_["dlc_count"]).clip(lower=0.0).values
        log1p_dlc = np.log1p(clean_dlc)

        raw_achs = pd.to_numeric(X.get("achievements", pd.Series(index=X.index)), errors="coerce")
        clean_achs = raw_achs.fillna(self.learned_medians_["achievements"]).clip(lower=0.0).values
        log1p_achs = np.log1p(clean_achs)

        # 4. Plateformes
        def to_bool_array(col_name: str) -> np.ndarray:
            if col_name not in X.columns:
                return np.zeros(n_samples, dtype=float)
            col = X[col_name]
            return col.map(
                lambda v: 1.0 if str(v).strip().lower() in ("true", "1", "t", "yes") else 0.0
            ).values

        win = to_bool_array("windows")
        mac = to_bool_array("mac")
        lin = to_bool_array("linux")
        plat_count = win + mac + lin

        # 5. Date de sortie
        years = []
        sin_months = []
        cos_months = []
        default_yr = self.learned_medians_["release_year"]

        date_col = X.get("release_date", pd.Series(index=X.index))
        for d in date_col:
            dt = pd.to_datetime(d, errors="coerce")
            if pd.notna(dt) and 1990 <= dt.year <= 2030:
                years.append(float(dt.year))
                m = float(dt.month)
                sin_months.append(np.sin(2.0 * np.pi * m / 12.0))
                cos_months.append(np.cos(2.0 * np.pi * m / 12.0))
            else:
                years.append(default_yr)
                sin_months.append(0.0)
                cos_months.append(1.0)

        years_arr = np.array(years, dtype=float)
        sin_arr = np.array(sin_months, dtype=float)
        cos_arr = np.array(cos_months, dtype=float)

        # 6. Genres
        parsed_genres_list = [safe_parse_genres(g) for g in X.get("genres", pd.Series(index=X.index))]
        genre_count_arr = np.array([float(len(gl)) for gl in parsed_genres_list], dtype=float)

        # Indicateurs de genres clés
        flag_arrays = []
        for flag in CORE_GENRE_FLAGS:
            flag_col = np.array(
                [1.0 if flag in gl else 0.0 for gl in parsed_genres_list], dtype=float
            )
            flag_arrays.append(flag_col)

        # One-hot des top genres principaux
        primary_genres = [gl[0] if gl else "unknown" for gl in parsed_genres_list]
        one_hot_arrays = []
        for g in self.top_genres_:
            col_g = np.array([1.0 if pg == g else 0.0 for pg in primary_genres], dtype=float)
            one_hot_arrays.append(col_g)

        # Catégorie autre pour les genres non recensés dans le top K
        is_known = np.zeros(n_samples, dtype=bool)
        for g_arr in one_hot_arrays:
            is_known |= (g_arr == 1.0)
        other_col = (~is_known).astype(float)
        one_hot_arrays.append(other_col)

        # Assemblage des colonnes
        cols_to_stack = [
            log1p_price,
            is_free,
            clean_ages,
            log1p_dlc,
            log1p_achs,
            win,
            mac,
            lin,
            plat_count,
            years_arr,
            sin_arr,
            cos_arr,
            genre_count_arr,
            *flag_arrays,
            *one_hot_arrays,
        ]

        feature_matrix = np.column_stack(cols_to_stack)
        return feature_matrix

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        return np.array(self.feature_names_)
