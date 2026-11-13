"""api_prediction/repository.py - Persistance relationnelle PostgreSQL 18.6 et transactions atomiques.

Fournit :
- Gestion d'un pool de connexions borné et résilient (`DatabasePool`).
- Enregistrement idempotent des identités de modèles (`tracing.model_versions`).
- Transaction atomique stricte d'inférence (api_requests + 2N predictions) avec rollback intégral en cas de panne.
- Enregistrement des requêtes en erreur et bascule vers le journal local de secours en cas d'indisponibilité SQL.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import logging
import os
from queue import Empty, Queue
import time
from typing import Any, Dict, Iterator, List, Optional
from uuid import UUID, uuid4
import psycopg
from psycopg.types.json import Jsonb

from api_prediction.schemas import GamePredictionItem, InferenceGameInput
from api_prediction.telemetry import log_telemetry_fallback
from src.settings import settings

logger = logging.getLogger(__name__)


class DatabasePool:
    """Pool de connexions PostgreSQL borné."""

    def __init__(
        self,
        database_url: Optional[str] = None,
        max_connections: int = 5,
        timeout: float = 5.0,
    ):
        self.database_url = database_url or settings.DATABASE_URL
        self.max_connections = max_connections
        self.timeout = timeout
        self._pool: Queue[psycopg.Connection] = Queue(maxsize=max_connections)
        self._created_connections = 0

    def _create_connection(self) -> psycopg.Connection:
        """Crée une nouvelle connexion avec configuration d'encodage et de schéma."""
        conn = psycopg.connect(self.database_url, autocommit=True)
        return conn

    @contextmanager
    def connection(self) -> Iterator[psycopg.Connection]:
        """Obtient une connexion disponible du pool avec libération automatique."""
        conn: Optional[psycopg.Connection] = None
        acquired = False
        try:
            # 1. Tenter d'obtenir une connexion existante du pool
            try:
                conn = self._pool.get_nowait()
                acquired = True
            except Empty:
                # 2. Si le pool est vide et qu'on n'a pas atteint la limite max
                if self._created_connections < self.max_connections:
                    conn = self._create_connection()
                    self._created_connections += 1
                    acquired = True
                else:
                    # 3. Attendre avec timeout qu'une connexion se libère
                    conn = self._pool.get(timeout=self.timeout)
                    acquired = True

            # Vérification de l'état de la connexion
            if conn.closed:
                conn = self._create_connection()

            yield conn

        except Exception as exc:
            if conn and not conn.closed:
                try:
                    conn.rollback()
                except Exception:
                    pass
            raise exc
        finally:
            if acquired and conn and not conn.closed:
                try:
                    self._pool.put_nowait(conn)
                except Exception:
                    conn.close()

    def is_healthy(self) -> bool:
        """Vérifie l'accessibilité réelle de PostgreSQL via SELECT 1."""
        try:
            with self.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1;")
                    return cur.fetchone()[0] == 1
        except Exception as exc:
            logger.warning(f"[DB HEALTH] Sonde PostgreSQL en échec : {exc}")
            return False

    def close_all(self) -> None:
        """Ferme toutes les connexions du pool."""
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                if not conn.closed:
                    conn.close()
            except Exception:
                pass


# Instance singleton du pool pour le runtime
db_pool = DatabasePool()


def register_or_get_model_version(
    pool: DatabasePool,
    task: str,
    tracking_uri: str,
    registered_name: str,
    version: int,
    mlflow_run_id: str,
    model_uri: str,
) -> UUID:
    """Enregistre ou retrouve l'UUID immuable d'un modèle dans tracing.model_versions."""
    with pool.connection() as conn:
        with conn.cursor() as cur:
            # Recherche de l'identité exacte
            cur.execute(
                """
                SELECT model_id, task, mlflow_run_id, model_uri
                FROM tracing.model_versions
                WHERE tracking_uri = %s AND registered_name = %s AND version = %s;
                """,
                (tracking_uri, registered_name, version),
            )
            row = cur.fetchone()
            if row:
                existing_id, ex_task, ex_run_id, ex_uri = row
                if ex_task != task:
                    raise ValueError(
                        f"Incohérence de tâche pour {registered_name} v{version} : {ex_task} != {task}"
                    )
                logger.info(
                    f"[DB REPOSITORY] Version existante retrouvée : {registered_name} v{version} -> {existing_id}"
                )
                return UUID(str(existing_id))

            # Création d'un nouvel identifiant
            new_id = uuid4()
            cur.execute(
                """
                INSERT INTO tracing.model_versions (
                    model_id, task, tracking_uri, registered_name, version, mlflow_run_id, model_uri
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (tracking_uri, registered_name, version) DO NOTHING;
                """,
                (new_id, task, tracking_uri, registered_name, version, mlflow_run_id, model_uri),
            )
            conn.commit()

            # Relire au cas où un autre worker l'aurait inséré en parallèle
            cur.execute(
                """
                SELECT model_id FROM tracing.model_versions
                WHERE tracking_uri = %s AND registered_name = %s AND version = %s;
                """,
                (tracking_uri, registered_name, version),
            )
            resolved_id = cur.fetchone()[0]
            logger.info(
                f"[DB REPOSITORY] Version enregistrée avec succès : {registered_name} v{version} -> {resolved_id}"
            )
            return UUID(str(resolved_id))


def record_prediction_transaction(
    pool: DatabasePool,
    request_id: UUID,
    started_at: datetime,
    endpoint: str,
    item_count: int,
    processing_duration_ms: float,
    inference_duration_ms: float,
    worker_pid: int,
    instance_id: UUID,
    api_revision: str,
    campaign_id: str,
    results: List[GamePredictionItem],
    games: List[InferenceGameInput],
    reg_model_id: UUID,
    cls_model_id: UUID,
) -> None:
    """Exécute la transaction SQL atomique stricte : api_requests + 2N predictions.

    RÈGLE ABSOLUE :
    Toute exception déclenche un ROLLBACK intégral. Aucune confirmation 200 n'est
    délivrée sans confirmation du COMMIT.
    """
    with pool.connection() as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                # 1. Insertion de la requête dans tracing.api_requests
                cur.execute(
                    """
                    INSERT INTO tracing.api_requests (
                        request_id, started_at, endpoint, http_status, item_count,
                        processing_duration_ms, inference_duration_ms, worker_pid,
                        instance_id, api_revision, campaign_id, error_code
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    );
                    """,
                    (
                        request_id,
                        started_at,
                        endpoint,
                        200,
                        item_count,
                        processing_duration_ms,
                        inference_duration_ms,
                        worker_pid,
                        instance_id,
                        api_revision,
                        campaign_id,
                        None,
                    ),
                )

                # 2. Insertion des 2N lignes de prédictions dans tracing.predictions
                for idx, game in enumerate(games):
                    item = results[idx]
                    features_dict = {
                        "price": game.price,
                        "required_age": game.required_age,
                        "dlc_count": game.dlc_count,
                        "achievements": game.achievements,
                        "windows": game.windows,
                        "mac": game.mac,
                        "linux": game.linux,
                        "genres": game.genres,
                        "release_date": game.release_date,
                    }

                    # Récupération des deux identifiants de prédictions
                    reg_pred_item = next(p for p in item.predictions if p.task == "regression")
                    cls_pred_item = next(p for p in item.predictions if p.task == "classification")

                    # Ligne 1 : Régression
                    cur.execute(
                        """
                        INSERT INTO tracing.predictions (
                            prediction_id, request_id, item_index, appid, task,
                            model_id, input_features, predicted_pct, predicted_owners
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s
                        );
                        """,
                        (
                            UUID(str(reg_pred_item.prediction_id)),
                            request_id,
                            idx,
                            game.appid,
                            "regression",
                            reg_model_id,
                            Jsonb(features_dict),
                            item.predicted_pct,
                            None,
                        ),
                    )

                    # Ligne 2 : Classification
                    cur.execute(
                        """
                        INSERT INTO tracing.predictions (
                            prediction_id, request_id, item_index, appid, task,
                            model_id, input_features, predicted_pct, predicted_owners
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s
                        );
                        """,
                        (
                            UUID(str(cls_pred_item.prediction_id)),
                            request_id,
                            idx,
                            game.appid,
                            "classification",
                            cls_model_id,
                            Jsonb(features_dict),
                            None,
                            item.predicted_owners,
                        ),
                    )


def record_api_request_error(
    pool: DatabasePool,
    request_id: UUID,
    started_at: datetime,
    endpoint: str,
    http_status: int,
    error_code: str,
    processing_duration_ms: float,
    worker_pid: int,
    instance_id: UUID,
    api_revision: str,
    campaign_id: str,
    item_count: Optional[int] = None,
) -> None:
    """Enregistre une trace d'erreur HTTP (4xx / 5xx) dans tracing.api_requests ou bascule en secours."""
    try:
        with pool.connection() as conn:
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO tracing.api_requests (
                            request_id, started_at, endpoint, http_status, item_count,
                            processing_duration_ms, inference_duration_ms, worker_pid,
                            instance_id, api_revision, campaign_id, error_code
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        );
                        """,
                        (
                            request_id,
                            started_at,
                            endpoint,
                            http_status,
                            item_count,
                            processing_duration_ms,
                            None,
                            worker_pid,
                            instance_id,
                            api_revision,
                            campaign_id,
                            error_code,
                        ),
                    )
    except Exception as exc:
        logger.warning(
            f"[DB REPOSITORY] Échec de l'enregistrement de l'erreur dans PostgreSQL : {exc}. Bascule locale."
        )
        log_telemetry_fallback(
            request_id=str(request_id),
            endpoint=endpoint,
            http_status=http_status,
            error_code=error_code,
            message=str(exc),
        )
