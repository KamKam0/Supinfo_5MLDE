"""src/preprocessing.py - Fonction mère 2 : preprocess_data.

Applique les règles d'éligibilité métier distinctes pour chaque tâche,
découpe les données selon les partitions sans fuite figées
et construit les TaskDatasets pour la régression et la classification.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
import pandas as pd

from src.contracts import IngestionResult, PreparedData, RunContext, TaskDataset
from src.exceptions import DataValidationError
from src.partitions import (
    build_stratified_partitions,
    load_reserved_appids,
    save_partitions_to_json,
    verify_partitions_integrity,
)
from src.settings import settings


def get_or_create_partitions(df: pd.DataFrame, partitions_dir: Path) -> Dict[str, List[int]]:
    """Charge les partitions existantes ou les génère de manière déterministe."""
    train_file = partitions_dir / "train_appids.json"
    val_file = partitions_dir / "val_appids.json"
    test_file = partitions_dir / "test_appids.json"

    if train_file.exists() and val_file.exists() and test_file.exists():
        with open(train_file, "r", encoding="utf-8") as f:
            train_ids = json.load(f)
        with open(val_file, "r", encoding="utf-8") as f:
            val_ids = json.load(f)
        with open(test_file, "r", encoding="utf-8") as f:
            test_ids = json.load(f)

        # Vérification d'étanchéité stricte anti-fuite entre Train, Val et Test
        verify_partitions_integrity(train_ids, val_ids, test_ids, total_expected=None)

        # Si le DataFrame contient de nouveaux jeux (PULL ou PUSH) non encore partitionnés
        existing_partitioned = set(train_ids) | set(val_ids) | set(test_ids)
        current_appids = set(df["appid"].dropna().astype(int))
        new_appids = current_appids - existing_partitioned

        if new_appids:
            reserved_set = set(val_ids) | set(test_ids)
            # Règle anti-fuite absolue : les nouveaux jeux ne doivent pas appartenir à la réserve
            eligible_for_train = sorted(new_appids - reserved_set)
            train_ids = sorted(set(train_ids) | set(eligible_for_train))
            print(
                f"[PREPROCESSING] {len(eligible_for_train)} nouveaux jeux intégrés au split Train "
                f"(aucun dans la réserve d'évaluation)."
            )

        reserved_ids = sorted(set(val_ids) | set(test_ids))
        return {
            "train_appids": train_ids,
            "val_appids": val_ids,
            "test_appids": test_ids,
            "reserved_appids": reserved_ids,
        }

    partitions = build_stratified_partitions(df, seed=settings.RANDOM_SEED)
    save_partitions_to_json(partitions, partitions_dir)
    return partitions


def preprocess_data(ingestion_result: IngestionResult) -> PreparedData:
    """Seconde fonction mère : filtrage d'éligibilité et découpage sans fuite."""
    if not ingestion_result.gx_success:
        raise DataValidationError(
            "[PREPROCESSING] Rejet bloquant : interdiction absolue de prétraiter des données "
            "non validées par Great Expectations."
        )

    snapshot_path = Path(ingestion_result.snapshot.path)
    df = pd.read_csv(snapshot_path)
    partitions_dir = settings.PROJECT_ROOT / "data" / "partitions"

    partitions = get_or_create_partitions(df, partitions_dir)
    train_appids_set = set(partitions["train_appids"])
    val_appids_set = set(partitions["val_appids"])
    test_appids_set = set(partitions["test_appids"])

    feature_cols = list(settings.ALLOWED_INFERENCE_FEATURES)

    # --------------------------------------------------------------------------
    # 1. Éligibilité Tâche Régression (pct_pos_total)
    # --------------------------------------------------------------------------
    reg_mask = (
        (df["pct_pos_total"] >= settings.REGRESSION_MIN_PCT)
        & (df["pct_pos_total"] <= settings.REGRESSION_MAX_PCT)
        & (df["num_reviews_total"] >= settings.REGRESSION_MIN_REVIEWS)
    )
    reg_df = df[reg_mask].copy()

    reg_train = sorted(set(reg_df["appid"]) & train_appids_set)
    reg_val = sorted(set(reg_df["appid"]) & val_appids_set)
    reg_test = sorted(set(reg_df["appid"]) & test_appids_set)

    regression_dataset = TaskDataset(
        task_name="regression",
        train_appids=reg_train,
        val_appids=reg_val,
        test_appids=reg_test,
        feature_names=feature_cols,
        target_name=settings.TARGET_REGRESSION,
    )

    # --------------------------------------------------------------------------
    # 2. Éligibilité Tâche Classification (estimated_owners)
    # --------------------------------------------------------------------------
    cls_df_non_zero = df[df["estimated_owners"] != settings.CLASSIFICATION_EXCLUDED_CLASS].copy()
    counts = cls_df_non_zero["estimated_owners"].value_counts()
    valid_classes = set(counts[counts >= settings.CLASSIFICATION_MIN_CLASS_COUNT].index)
    cls_df = cls_df_non_zero[cls_df_non_zero["estimated_owners"].isin(valid_classes)].copy()

    cls_train = sorted(set(cls_df["appid"]) & train_appids_set)
    cls_val = sorted(set(cls_df["appid"]) & val_appids_set)
    cls_test = sorted(set(cls_df["appid"]) & test_appids_set)

    classification_dataset = TaskDataset(
        task_name="classification",
        train_appids=cls_train,
        val_appids=cls_val,
        test_appids=cls_test,
        feature_names=feature_cols,
        target_name=settings.TARGET_CLASSIFICATION,
    )

    print(
        f"[PREPROCESSING] Régression éligible : {len(reg_df):,} jeux "
        f"(Train: {len(reg_train):,}, Val: {len(reg_val):,}, Test: {len(reg_test):,})"
    )
    print(
        f"[PREPROCESSING] Classification éligible : {len(cls_df):,} jeux ({len(valid_classes)} classes) "
        f"(Train: {len(cls_train):,}, Val: {len(cls_val):,}, Test: {len(cls_test):,})"
    )

    return PreparedData(
        context=ingestion_result.context,
        regression_data=regression_dataset,
        classification_data=classification_dataset,
        transformers_info={"feature_builder": "SteamFeatureBuilder"},
    )
