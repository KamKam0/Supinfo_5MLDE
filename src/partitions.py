"""src/partitions.py - Protocole de partitionnement déterministe et barrières anti-fuite.

Gère le découpage Train (70%), Qualification/Validation (15%), Test final (15%)
par stratification sur les identifiants uniques appid, et impose la vérification
stricte d'étanchéité anti-fuite (data leakage).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def build_stratified_partitions(
    df: pd.DataFrame,
    seed: int = 42,
    stratify_col: str = "estimated_owners",
    test_eval_ratio: float = 0.30,
    val_test_ratio: float = 0.50,
) -> Dict[str, List[int]]:
    """Génère les listes d'appids pour Train, Val et Test avec stratification robuste."""
    if "appid" not in df.columns:
        raise ValueError("La colonne 'appid' est obligatoire pour partitionner les jeux.")

    # Copie de la colonne de stratification avec traitement des classes rares
    strat_series = df[stratify_col].fillna("missing").astype(str).copy()
    counts = strat_series.value_counts()
    rare_categories = counts[counts < 5].index
    strat_series = strat_series.replace(rare_categories, "__rare_grouped__")

    # 1. Découpage Train (70%) vs Évaluation (30%)
    train_df, eval_df = train_test_split(
        df,
        test_size=test_eval_ratio,
        stratify=strat_series,
        random_state=seed,
    )

    # 2. Découpage Évaluation en Validation (15%) et Test (15%)
    eval_strat = strat_series.loc[eval_df.index]
    val_df, test_df = train_test_split(
        eval_df,
        test_size=val_test_ratio,
        stratify=eval_strat,
        random_state=seed,
    )

    train_ids = sorted(train_df["appid"].astype(int).tolist())
    val_ids = sorted(val_df["appid"].astype(int).tolist())
    test_ids = sorted(test_df["appid"].astype(int).tolist())
    reserved_ids = sorted(set(val_ids) | set(test_ids))

    # Vérification d'intégrité immédiate
    verify_partitions_integrity(train_ids, val_ids, test_ids, len(df))

    return {
        "train_appids": train_ids,
        "val_appids": val_ids,
        "test_appids": test_ids,
        "reserved_appids": reserved_ids,
    }


def verify_partitions_integrity(
    train_ids: List[int], val_ids: List[int], test_ids: List[int], total_expected: int | None = None
) -> bool:
    """Vérifie mathématiquement qu'aucune intersection n'existe entre les partitions."""
    s_train = set(train_ids)
    s_val = set(val_ids)
    s_test = set(test_ids)

    inter_train_val = s_train & s_val
    inter_train_test = s_train & s_test
    inter_val_test = s_val & s_test

    if inter_train_val:
        raise ValueError(f"FUITE DÉTECTÉE : {len(inter_train_val)} identifiants communs entre Train et Val !")
    if inter_train_test:
        raise ValueError(f"FUITE DÉTECTÉE : {len(inter_train_test)} identifiants communs entre Train et Test !")
    if inter_val_test:
        raise ValueError(f"FUITE DÉTECTÉE : {len(inter_val_test)} identifiants communs entre Val et Test !")

    if total_expected is not None:
        total_unique = len(s_train | s_val | s_test)
        if total_unique != total_expected:
            raise ValueError(
                f"Incohérence d'effectif : {total_unique} identifiants uniques trouvés, {total_expected} attendus."
            )

    return True


def save_partitions_to_json(partitions: Dict[str, List[int]], output_dir: Path | str) -> None:
    """Sauvegarde les listes d'appids dans des fichiers JSON indépendants."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for key, id_list in partitions.items():
        file_path = out_dir / f"{key}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(id_list, f, indent=2)


def load_reserved_appids(reserved_file_path: Path | str) -> Set[int]:
    """Charge l'ensemble des appids protégés contre toute injection d'apprentissage."""
    p = Path(reserved_file_path)
    if not p.exists():
        return set()
    with open(p, "r", encoding="utf-8") as f:
        return set(json.load(f))


def filter_out_reserved_games(df: pd.DataFrame, reserved_ids: Set[int]) -> pd.DataFrame:
    """Filtre formellement tout lot entrant pour exclure les appids réservés (règle anti-leakage)."""
    if not reserved_ids or "appid" not in df.columns:
        return df
    mask = ~df["appid"].isin(reserved_ids)
    excluded_count = (~mask).sum()
    if excluded_count > 0:
        print(f"[ANTI-LEAKAGE] {excluded_count} jeux exclus car présents dans la réserve de qualification/test.")
    return df[mask].copy()
