"""api_prediction/service.py - Moteur de prédiction conjointe vectorisée.

Exécute les inférences en mémoire sur les deux pipelines pré-chargés :
- Conversion vectorisée des entrées Pydantic en DataFrame pandas.
- Appel unique par tâche pour l'ensemble du lot (vectorisation complète).
- Contrôle strict des sorties et génération d'identifiants de prédiction immuables.
"""

from __future__ import annotations

import logging
import time
from typing import List, Tuple
from uuid import uuid4
import numpy as np
import pandas as pd

from api_prediction.lifecycle import ModelServiceState
from api_prediction.schemas import (
    GamePredictionItem,
    InferenceGameInput,
    TaskModelInfo,
    TaskPredictionResult,
)

logger = logging.getLogger(__name__)


class ServiceNotReadyError(Exception):
    """Exception levée lorsque le worker n'a pas validé son cycle de démarrage."""
    pass


class ModelContractError(Exception):
    """Exception levée lorsqu'une inférence viole le contrat de sortie."""
    pass


def games_to_dataframe(games: List[InferenceGameInput]) -> pd.DataFrame:
    """Convertit une liste de jeux conformes en DataFrame optimisé pour scikit-learn."""
    rows = []
    for g in games:
        row = {
            "price": g.price,
            "required_age": g.required_age,
            "dlc_count": g.dlc_count,
            "achievements": g.achievements,
            "windows": g.windows,
            "mac": g.mac,
            "linux": g.linux,
            "genres": g.genres,
            "release_date": g.release_date,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def run_prediction_batch(
    games: List[InferenceGameInput],
    state: ModelServiceState,
    server_uri: str,
) -> Tuple[List[GamePredictionItem], float]:
    """Exécute l'inférence vectorisée conjointe et structure la réponse contractuelle."""
    if not state.is_ready or state.reg_model is None or state.cls_model is None:
        reason = state.ready_error or "Modèles ML non instanciés en mémoire."
        raise ServiceNotReadyError(f"Le service d'inférence n'est pas opérationnel : {reason}")

    if not games:
        raise ValueError("La liste de jeux soumise à l'inférence est vide.")

    df = games_to_dataframe(games)
    n_items = len(games)

    # Inférence conjointe vectorisée avec chronométrage haute précision
    t0 = time.perf_counter()
    try:
        reg_preds = state.reg_model.predict(df)
        cls_preds = state.cls_model.predict(df)
    except Exception as exc:
        logger.error(f"[PREDICTION] Erreur d'exécution du pipeline de calcul : {exc}")
        raise ModelContractError(f"Erreur d'exécution du modèle : {exc}") from exc
    inference_duration_ms = (time.perf_counter() - t0) * 1000.0

    # Vérification d'invariance de dimension
    if len(reg_preds) != n_items or len(cls_preds) != n_items:
        raise ModelContractError(
            f"Incohérence de cardinalité : {n_items} entrées, {len(reg_preds)} régressions, "
            f"{len(cls_preds)} classifications."
        )

    # Construction des métadonnées de modèles
    reg_meta = state.reg_info
    cls_meta = state.cls_info

    reg_model_info = TaskModelInfo(
        server_uri=server_uri,
        model_name=reg_meta.get("model_name", "steam-positive-ratio"),
        model_version=str(reg_meta.get("version", "1")),
        mlflow_run_id=reg_meta.get("mlflow_run_id", "unknown"),
    )

    cls_model_info = TaskModelInfo(
        server_uri=server_uri,
        model_name=cls_meta.get("model_name", "steam-owner-range"),
        model_version=str(cls_meta.get("version", "1")),
        mlflow_run_id=cls_meta.get("mlflow_run_id", "unknown"),
    )

    results: List[GamePredictionItem] = []

    for idx, game in enumerate(games):
        raw_reg = float(reg_preds[idx])
        # Règle de bornage [0, 100]
        clipped_reg = max(0.0, min(100.0, raw_reg))
        clean_reg = round(clipped_reg, 4)

        raw_cls = str(cls_preds[idx]).strip()

        pred_reg_id = str(uuid4())
        pred_cls_id = str(uuid4())

        task_reg = TaskPredictionResult(
            prediction_id=pred_reg_id,
            task="regression",
            target_column="pct_pos_total",
            predicted_value=clean_reg,
            unit="pct_pos_total [0, 100]",
            model=reg_model_info,
        )

        task_cls = TaskPredictionResult(
            prediction_id=pred_cls_id,
            task="classification",
            target_column="estimated_owners",
            predicted_value=raw_cls,
            unit="class_label",
            model=cls_model_info,
        )

        item = GamePredictionItem(
            item_index=idx,
            appid=game.appid,
            predictions=[task_reg, task_cls],
            predicted_pct=clean_reg,
            predicted_owners=raw_cls,
        )
        results.append(item)

    return results, inference_duration_ms
