"""src/orchestration/cache_policy.py - Politique de mise en cache déterministe des calculs purs.

Fournit :
- Calcul de clé composite infalsifiable :
  clé = SHA-256(SHA-256 du snapshot + paramètres JSON normalisés + SHA-256 du code source).
- Gestionnaire de cache local persistant (FeatureCache) avec traçabilité des hits/misses.
- Fonction pure de construction de features avec réutilisation instantanée et invalidation stricte.
- Tâche Prefect 3 décorée intégrant la politique de cache sur calcul pur sans fuite d'état.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import pandas as pd
from prefect import task

from src.feature_builder import SteamFeatureBuilder
from src.settings import settings

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = settings.PROJECT_ROOT / "data" / "cache" / "features"


def compute_code_hash(source_obj: Any = None) -> str:
    """Calcule l'empreinte SHA-256 du code source d'un module ou d'une classe."""
    target = source_obj if source_obj is not None else SteamFeatureBuilder
    try:
        source_text = inspect.getsource(target)
    except Exception:
        # Repli sur le fichier source si l'inspection directe échoue
        source_file = Path(inspect.getfile(target))
        with open(source_file, "r", encoding="utf-8") as f:
            source_text = f.read()

    return hashlib.sha256(source_text.encode("utf-8")).hexdigest().lower()


def compute_dataframe_content_hash(df: pd.DataFrame) -> str:
    """Calcule une empreinte SHA-256 déterministe et invariante de l'état d'un DataFrame."""
    # Hachage rapide et robuste fondé sur les colonnes, le schéma et les valeurs
    col_str = ",".join(sorted(df.columns))
    row_count = len(df)
    sample_records = df.head(100).to_json(orient="values", date_format="iso")
    tail_records = df.tail(100).to_json(orient="values", date_format="iso")
    raw = f"{col_str}|{row_count}|{sample_records}|{tail_records}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest().lower()


def compute_feature_cache_key(
    content_sha256: str,
    parameters: Dict[str, Any],
    code_hash: Optional[str] = None,
) -> str:
    """Construit la clé composite officielle de mise en cache.

    clé = SHA-256(content_sha256 + normalized_parameters_json + code_hash)
    """
    actual_code_hash = code_hash or compute_code_hash()
    norm_params = json.dumps(parameters, sort_keys=True, ensure_ascii=True)
    composite = f"content:{content_sha256}|params:{norm_params}|code:{actual_code_hash}"
    return hashlib.sha256(composite.encode("utf-8")).hexdigest().lower()


class FeatureCache:
    """Gestionnaire persistant du cache local sur disque."""

    def __init__(self, cache_dir: Optional[Path | str] = None) -> None:
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits_count: int = 0
        self.misses_count: int = 0

    def _get_cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.parquet"

    def get(self, key: str) -> Optional[pd.DataFrame]:
        """Récupère un DataFrame depuis le cache s'il existe."""
        p = self._get_cache_path(key)
        if p.exists():
            try:
                df = pd.read_parquet(p)
                self.hits_count += 1
                logger.info(f"[CACHE HIT] Features réutilisées depuis le cache : {p.name} ({len(df):,} lignes)")
                return df
            except Exception as exc:
                logger.warning(f"[CACHE] Échec de lecture du fichier de cache {p.name} : {exc}")
        self.misses_count += 1
        return None

    def set(self, key: str, df: pd.DataFrame) -> Path:
        """Enregistre un DataFrame dans le cache."""
        p = self._get_cache_path(key)
        df.to_parquet(p, index=False)
        logger.info(f"[CACHE SAVE] Features persistées dans le cache : {p.name} ({len(df):,} lignes)")
        return p

    def clear(self) -> int:
        """Supprime tous les fichiers de cache."""
        deleted = 0
        for f in self.cache_dir.glob("*.parquet"):
            try:
                f.unlink()
                deleted += 1
            except Exception:
                pass
        return deleted

    def stats(self) -> Dict[str, Any]:
        """Retourne les métriques opérationnelles du cache."""
        entries = list(self.cache_dir.glob("*.parquet"))
        total_size = sum(f.stat().st_size for f in entries)
        return {
            "entries_count": len(entries),
            "total_bytes": total_size,
            "hits": self.hits_count,
            "misses": self.misses_count,
            "cache_dir": str(self.cache_dir),
        }


def build_features_pure(
    df: pd.DataFrame,
    top_genres_k: int = 15,
    default_year: float = 2020.0,
    content_sha256: Optional[str] = None,
    cache_dir: Optional[Path | str] = None,
    force_recompute: bool = False,
    custom_code_hash: Optional[str] = None,
) -> Tuple[pd.DataFrame, bool, str]:
    """Fonction pure de construction des features avec mise en cache déterministe.

    Returns:
        Tuple (features_df, cache_hit, cache_key)
    """
    actual_content_hash = content_sha256 or compute_dataframe_content_hash(df)
    params = {"top_genres_k": top_genres_k, "default_year": default_year}
    code_hash = custom_code_hash or compute_code_hash()

    cache_key = compute_feature_cache_key(
        content_sha256=actual_content_hash,
        parameters=params,
        code_hash=code_hash,
    )

    cache = FeatureCache(cache_dir=cache_dir)

    if not force_recompute:
        cached_df = cache.get(cache_key)
        if cached_df is not None:
            return cached_df, True, cache_key

    # Cache MISS : Calcul pur et déterministe
    logger.info(
        f"[CACHE MISS] Calcul pur des caractéristiques (clé: {cache_key[:12]}..., top_genres_k: {top_genres_k})"
    )
    builder = SteamFeatureBuilder(top_genres_k=top_genres_k, default_year=default_year)
    transformed_matrix = builder.fit_transform(df)

    # Conversion en DataFrame standardisé pour sérialisation
    feature_names = builder.get_feature_names_out()
    features_df = pd.DataFrame(transformed_matrix, columns=feature_names)

    # Persistance dans le cache
    cache.set(cache_key, features_df)
    return features_df, False, cache_key


# ==============================================================================
# TÂCHE PREFECT AVEC CLÉ DE CACHE COMPOSITE
# ==============================================================================


@task(
    name="task_build_features_pure",
    description="Construction déterministe et isolée des caractéristiques avec cache de calcul pur",
    retries=0,
)
def task_build_features_pure(
    df: pd.DataFrame,
    top_genres_k: int = 15,
    default_year: float = 2020.0,
    content_sha256: Optional[str] = None,
    cache_dir: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Tâche Prefect effectuant le feature engineering pur sous protection de cache."""
    features_df, cache_hit, key = build_features_pure(
        df=df,
        top_genres_k=top_genres_k,
        default_year=default_year,
        content_sha256=content_sha256,
        cache_dir=cache_dir,
    )
    status_str = "HIT" if cache_hit else "MISS"
    logger.info(f"[TASK:BUILD_FEATURES] Statut cache: {status_str} (clé={key[:12]}...)")
    return features_df
