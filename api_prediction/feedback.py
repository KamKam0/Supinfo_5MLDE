"""api_prediction/feedback.py - Gestionnaire transactionnel idempotent des feedbacks.

Gère l'endpoint POST /feedback avec respect strict des règles de concurrence :
- 201 Created : Premier enregistrement conforme.
- 200 OK : Rejeu idempotent (même feedback_id, même contenu).
- 409 Conflict : Conflit d'identifiant (même feedback_id avec contenu distinct) ou tentative d'annoter une prédiction déjà renseignée.
- 404 Not Found : Prédiction inconnue dans tracing.predictions.
- 422 Unprocessable Entity : Incompatibilité entre la tâche de la prédiction et l'observation reçue.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import time
from typing import Any, Dict, Optional, Tuple
from uuid import UUID

from api_prediction.repository import DatabasePool, record_api_request_error
from api_prediction.schemas import FeedbackInput

logger = logging.getLogger(__name__)


def process_feedback_transaction(
    pool: DatabasePool,
    request_id: UUID,
    started_at: datetime,
    worker_pid: int,
    instance_id: UUID,
    api_revision: str,
    campaign_id: str,
    feedback: FeedbackInput,
) -> Tuple[int, Dict[str, Any]]:
    """Traite la réception d'un retour utilisateur/observation dans une transaction PostgreSQL."""
    f_id = UUID(str(feedback.feedback_id))
    p_id = UUID(str(feedback.prediction_id))

    t0 = time.perf_counter()

    with pool.connection() as conn:
        with conn.cursor() as cur:
            # 1. Vérifier si la prédiction existe dans tracing.predictions
            cur.execute(
                """
                SELECT prediction_id, task, appid
                FROM tracing.predictions
                WHERE prediction_id = %s;
                """,
                (p_id,),
            )
            pred_row = cur.fetchone()
            if not pred_row:
                proc_dur = (time.perf_counter() - t0) * 1000.0
                record_api_request_error(
                    pool=pool,
                    request_id=request_id,
                    started_at=started_at,
                    endpoint="/feedback",
                    http_status=404,
                    error_code="PREDICTION_NOT_FOUND",
                    processing_duration_ms=proc_dur,
                    worker_pid=worker_pid,
                    instance_id=instance_id,
                    api_revision=api_revision,
                    campaign_id=campaign_id,
                )
                return 404, {
                    "request_id": str(request_id),
                    "error_code": "PREDICTION_NOT_FOUND",
                    "message": f"La prédiction '{p_id}' n'existe pas dans le registre.",
                }

            pred_task = str(pred_row[1])

            # 2. Vérification de cohérence de tâche
            if pred_task == "regression" and feedback.observed_owners is not None:
                proc_dur = (time.perf_counter() - t0) * 1000.0
                record_api_request_error(
                    pool=pool,
                    request_id=request_id,
                    started_at=started_at,
                    endpoint="/feedback",
                    http_status=422,
                    error_code="INVALID_INPUT",
                    processing_duration_ms=proc_dur,
                    worker_pid=worker_pid,
                    instance_id=instance_id,
                    api_revision=api_revision,
                    campaign_id=campaign_id,
                )
                return 422, {
                    "request_id": str(request_id),
                    "error_code": "INVALID_INPUT",
                    "message": "Incohérence : une prédiction de régression ne peut pas recevoir d'observation de tranche 'observed_owners'.",
                }

            if pred_task == "classification" and feedback.observed_pct is not None:
                proc_dur = (time.perf_counter() - t0) * 1000.0
                record_api_request_error(
                    pool=pool,
                    request_id=request_id,
                    started_at=started_at,
                    endpoint="/feedback",
                    http_status=422,
                    error_code="INVALID_INPUT",
                    processing_duration_ms=proc_dur,
                    worker_pid=worker_pid,
                    instance_id=instance_id,
                    api_revision=api_revision,
                    campaign_id=campaign_id,
                )
                return 422, {
                    "request_id": str(request_id),
                    "error_code": "INVALID_INPUT",
                    "message": "Incohérence : une prédiction de classification ne peut pas recevoir d'observation de pourcentage 'observed_pct'.",
                }

            # 3. Vérifier si un feedback existe déjà avec cet identifiant feedback_id
            cur.execute(
                """
                SELECT feedback_id, request_id, prediction_id, task, useful, observed_pct, observed_owners, label_source
                FROM tracing.feedbacks
                WHERE feedback_id = %s;
                """,
                (f_id,),
            )
            existing_fb = cur.fetchone()

            if existing_fb:
                # Analyse d'idempotence
                ex_pred_id = existing_fb[2]
                ex_task = existing_fb[3]
                ex_useful = existing_fb[4]
                ex_pct = existing_fb[5]
                ex_owners = (existing_fb[6] or "").strip()
                ex_source = (existing_fb[7] or "").strip()

                same_pred = UUID(str(ex_pred_id)) == p_id
                same_task = ex_task == pred_task
                same_useful = ex_useful == feedback.useful

                same_pct = (ex_pct is None and feedback.observed_pct is None) or (
                    ex_pct is not None
                    and feedback.observed_pct is not None
                    and abs(float(ex_pct) - float(feedback.observed_pct)) < 1e-4
                )
                same_owners = ex_owners == (feedback.observed_owners or "").strip()
                same_source = ex_source == (feedback.label_source or "").strip()

                if same_pred and same_task and same_useful and same_pct and same_owners and same_source:
                    # Cas : Rejeu idempotent (200 OK)
                    proc_dur = (time.perf_counter() - t0) * 1000.0
                    with conn.transaction():
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
                                "/feedback",
                                200,
                                None,
                                proc_dur,
                                None,
                                worker_pid,
                                instance_id,
                                api_revision,
                                campaign_id,
                                None,
                            ),
                        )
                    return 200, {
                        "request_id": str(request_id),
                        "feedback_id": str(f_id),
                        "prediction_id": str(p_id),
                        "status": "replayed",
                        "message": "Feedback identique déjà enregistré (rejeu idempotent).",
                    }
                else:
                    # Cas : Conflit d'identifiant avec contenu différent (409 Conflict)
                    proc_dur = (time.perf_counter() - t0) * 1000.0
                    record_api_request_error(
                        pool=pool,
                        request_id=request_id,
                        started_at=started_at,
                        endpoint="/feedback",
                        http_status=409,
                        error_code="FEEDBACK_CONFLICT",
                        processing_duration_ms=proc_dur,
                        worker_pid=worker_pid,
                        instance_id=instance_id,
                        api_revision=api_revision,
                        campaign_id=campaign_id,
                    )
                    return 409, {
                        "request_id": str(request_id),
                        "error_code": "FEEDBACK_CONFLICT",
                        "message": f"Un feedback portant l'ID '{f_id}' existe déjà mais avec un contenu différent.",
                    }

            # 4. Vérifier si la prédiction est déjà annotée par un autre feedback
            cur.execute(
                """
                SELECT feedback_id FROM tracing.feedbacks WHERE prediction_id = %s;
                """,
                (p_id,),
            )
            already_annotated = cur.fetchone()
            if already_annotated:
                proc_dur = (time.perf_counter() - t0) * 1000.0
                record_api_request_error(
                    pool=pool,
                    request_id=request_id,
                    started_at=started_at,
                    endpoint="/feedback",
                    http_status=409,
                    error_code="FEEDBACK_CONFLICT",
                    processing_duration_ms=proc_dur,
                    worker_pid=worker_pid,
                    instance_id=instance_id,
                    api_revision=api_revision,
                    campaign_id=campaign_id,
                )
                return 409, {
                    "request_id": str(request_id),
                    "error_code": "FEEDBACK_CONFLICT",
                    "message": f"La prédiction '{p_id}' a déjà reçu un feedback ({already_annotated[0]}). Unicité violée.",
                }

            # 5. Création conforme d'un nouveau feedback (201 Created)
            obs_dt: Optional[datetime] = None
            if feedback.observed_at:
                try:
                    obs_dt = datetime.fromisoformat(feedback.observed_at.replace("Z", "+00:00"))
                except Exception:
                    obs_dt = datetime.now(timezone.utc)

            proc_dur = (time.perf_counter() - t0) * 1000.0

            with conn.transaction():
                # Insertion dans api_requests
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
                        "/feedback",
                        201,
                        None,
                        proc_dur,
                        None,
                        worker_pid,
                        instance_id,
                        api_revision,
                        campaign_id,
                        None,
                    ),
                )

                # Insertion dans feedbacks
                cur.execute(
                    """
                    INSERT INTO tracing.feedbacks (
                        feedback_id, request_id, prediction_id, task, useful,
                        observed_pct, observed_owners, observed_at, label_source
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s
                    );
                    """,
                    (
                        f_id,
                        request_id,
                        p_id,
                        pred_task,
                        feedback.useful,
                        feedback.observed_pct,
                        feedback.observed_owners.strip() if feedback.observed_owners else None,
                        obs_dt,
                        feedback.label_source.strip() if feedback.label_source else None,
                    ),
                )

            return 201, {
                "request_id": str(request_id),
                "feedback_id": str(f_id),
                "prediction_id": str(p_id),
                "status": "created",
                "message": "Feedback enregistré avec succès.",
            }
