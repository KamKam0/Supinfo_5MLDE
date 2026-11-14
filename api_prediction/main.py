"""api_prediction/main.py - Application FastAPI d'Inférence Steam & Persistance Relationnelle (Port 8001).

Expose le service d'inférence en ligne haute performance et de gestion des retours :
- POST /predict : Inférence conjointe unitaire avec persistance transactionnelle atomique PostgreSQL.
- POST /predict/batch : Inférence vectorisée par lot (1 à 100 jeux) avec persistance des 2N prédictions.
- POST /feedback : Réception des retours utilisateurs et vérités terrain avec détection d'idempotence et de conflits.
- GET /health/live : Sonde de vivacité élémentaire (HTTP 200).
- GET /health/ready : Sonde d'aptitude opérationnelle vérifiant les modèles en mémoire et la base PostgreSQL (200 ou 503).
- GET /models : Consultation du manifeste actif et des versions de modèles figées.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
import time
from typing import Any, Union
from uuid import NAMESPACE_DNS, UUID, uuid4, uuid5

from fastapi import Body, FastAPI, HTTPException, Request, Response, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api_prediction.feedback import process_feedback_transaction
from api_prediction.lifecycle import ModelServiceState, lifespan
from api_prediction.repository import (
    db_pool,
    record_api_request_error,
    record_prediction_transaction,
)
from api_prediction.schemas import (
    BatchPredictionRequest,
    ErrorResponse,
    FeedbackInput,
    FeedbackResponse,
    HealthResponse,
    InferenceGameInput,
    ModelsInfoResponse,
    PredictionResponse,
    SinglePredictionRequest,
)
from api_prediction.service import (
    ModelContractError,
    ServiceNotReadyError,
    run_prediction_batch,
)
from src.settings import settings

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Steam MLOps Prediction Service",
    description=(
        "Service d'inférence en ligne haute performance pour l'estimation conjointe du pourcentage "
        "d'avis positifs et de la tranche de propriétaires estimés sur le catalogue Steam."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_request_ids(request: Request) -> tuple[str, UUID]:
    """Retourne le request_id textuel et son équivalent UUID PostgreSQL garanti."""
    req_id = getattr(request.state, "request_id", None) or str(uuid4())
    req_uuid = getattr(request.state, "request_uuid", None)
    if req_uuid is None:
        try:
            req_uuid = UUID(req_id)
        except Exception:
            req_uuid = uuid5(NAMESPACE_DNS, req_id)
    return req_id, req_uuid


@app.middleware("http")
async def correlation_and_headers_middleware(request: Request, call_next: Any) -> Response:
    """Injecte un identifiant de corrélation UUID universel et les en-têtes d'audit de versions."""
    raw_req_id = request.headers.get("X-Request-ID")
    if raw_req_id:
        try:
            req_uuid = UUID(raw_req_id)
            request_id = raw_req_id
        except ValueError:
            req_uuid = uuid5(NAMESPACE_DNS, raw_req_id)
            request_id = raw_req_id
    else:
        req_uuid = uuid4()
        request_id = str(req_uuid)

    request.state.request_id = request_id
    request.state.request_uuid = req_uuid
    request.state.started_at = datetime.now(timezone.utc)

    response: Response = await call_next(request)

    response.headers["X-Request-ID"] = request_id
    state: ModelServiceState = getattr(request.app.state, "service_state", None)
    if state:
        response.headers["X-Manifest-ID"] = state.manifest_id
        response.headers["X-Worker-Instance-ID"] = state.instance_id
        if state.reg_info:
            response.headers["X-Model-Regression-Version"] = str(
                state.reg_info.get("version", "unknown")
            )
        if state.cls_info:
            response.headers["X-Model-Classification-Version"] = str(
                state.cls_info.get("version", "unknown")
            )

    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Gestionnaire d'erreurs de validation 422 avec traçabilité et formatage sans fuite."""
    request_id, req_uuid = get_request_ids(request)
    started_at = getattr(request.state, "started_at", datetime.now(timezone.utc))
    state: ModelServiceState = getattr(request.app.state, "service_state", None)

    # Enregistrement de la trace d'erreur 422 si le service est initialisé
    if state and state.instance_id:
        try:
            record_api_request_error(
                pool=db_pool,
                request_id=req_uuid,
                started_at=started_at,
                endpoint=request.url.path,
                http_status=422,
                error_code="INVALID_INPUT",
                processing_duration_ms=0.0,
                worker_pid=os.getpid(),
                instance_id=UUID(state.instance_id),
                api_revision=state.manifest.get("application_revision", "unknown"),
                campaign_id=request.headers.get("X-Campaign-ID", "manual"),
            )
        except Exception:
            pass

    error_payload = ErrorResponse(
        request_id=request_id,
        error_code="INVALID_INPUT",
        message="Erreur de validation de la requête d'inférence (contrat strict sans variables cibles).",
        details=jsonable_encoder(exc.errors()),
    ).model_dump()

    return JSONResponse(
        status_code=422,
        content=error_payload,
        headers={"X-Request-ID": request_id},
    )


@app.exception_handler(ServiceNotReadyError)
async def service_not_ready_handler(request: Request, exc: ServiceNotReadyError) -> JSONResponse:
    """Gestionnaire d'indisponibilité du service 503."""
    request_id = getattr(request.state, "request_id", str(uuid4()))
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=ErrorResponse(
            request_id=request_id,
            error_code="DEPENDENCY_UNAVAILABLE",
            message=str(exc),
        ).model_dump(),
        headers={"X-Request-ID": request_id},
    )


@app.exception_handler(ModelContractError)
async def model_contract_handler(request: Request, exc: ModelContractError) -> JSONResponse:
    """Gestionnaire de rupture de contrat des modèles 503."""
    request_id = getattr(request.state, "request_id", str(uuid4()))
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=ErrorResponse(
            request_id=request_id,
            error_code="MODEL_CONTRACT_ERROR",
            message=str(exc),
        ).model_dump(),
        headers={"X-Request-ID": request_id},
    )


@app.get("/health/live", response_model=HealthResponse, tags=["Observabilité"])
def health_live() -> HealthResponse:
    """Sonde de vivacité élémentaire confirmant que le processus répond."""
    return HealthResponse(
        status="live",
        service="steam-prediction-api",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/health/ready", response_model=HealthResponse, tags=["Observabilité"])
def health_ready(request: Request, response: Response) -> HealthResponse:
    """Sonde d'aptitude opérationnelle (modèles en mémoire ET connectivité PostgreSQL requises)."""
    state: ModelServiceState = getattr(request.app.state, "service_state", None)
    now_iso = datetime.now(timezone.utc).isoformat()

    db_ok = db_pool.is_healthy()
    models_ok = state is not None and state.is_ready

    if not models_ok or not db_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        reason = []
        if not models_ok:
            reason.append(state.ready_error if state else "Worker non initialisé")
        if not db_ok:
            reason.append("Base de données PostgreSQL inaccessible")

        return HealthResponse(
            status="not_ready",
            service="steam-prediction-api",
            instance_id=state.instance_id if state else None,
            manifest_id=state.manifest_id if state else None,
            models_loaded=models_ok,
            timestamp=now_iso,
            error="; ".join(reason),
        )

    return HealthResponse(
        status="ready",
        service="steam-prediction-api",
        instance_id=state.instance_id,
        manifest_id=state.manifest_id,
        models_loaded=True,
        timestamp=now_iso,
    )


@app.get("/models", response_model=ModelsInfoResponse, tags=["Gouvernance"])
def get_models_info(request: Request) -> ModelsInfoResponse:
    """Consulte le manifeste de service et les versions exactes actives."""
    state: ModelServiceState = getattr(request.app.state, "service_state", None)
    if state is None or not state.manifest:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Manifeste de service non initialisé.",
        )

    return ModelsInfoResponse(
        manifest_id=state.manifest_id,
        instance_id=state.instance_id,
        mode=state.mode,
        generated_at=state.manifest.get("generated_at", ""),
        manifest_sha256=state.manifest_sha256,
        mlflow_tracking_uri=state.manifest.get(
            "mlflow_tracking_uri", settings.MLFLOW_TRACKING_URI
        ),
        models=state.manifest.get("models", {}),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inférence"],
    responses={
        422: {"model": ErrorResponse, "description": "Validation d'entrée rejetée (extra interdit)"},
        503: {"model": ErrorResponse, "description": "Modèles ou base de données indisponibles"},
    },
)
def predict_single(
    request: Request,
    payload: Union[SinglePredictionRequest, InferenceGameInput] = Body(...),
) -> PredictionResponse:
    """Inférence unitaire conjointe avec persistance relationnelle atomique."""
    t_req_start = time.perf_counter()
    request_id, req_uuid = get_request_ids(request)
    started_at = getattr(request.state, "started_at", datetime.now(timezone.utc))
    state: ModelServiceState = getattr(request.app.state, "service_state", None)

    if state is None or not state.is_ready:
        reason = state.ready_error if state else "Service non prêt"
        raise ServiceNotReadyError(f"Le service d'inférence n'est pas prêt : {reason}")

    # Extraction du jeu (support de '{"game": {...}}' ou direct '{...}')
    game_input = payload.game if isinstance(payload, SinglePredictionRequest) else payload

    server_uri = state.manifest.get("mlflow_tracking_uri", settings.MLFLOW_TRACKING_URI)
    results, inf_duration_ms = run_prediction_batch(
        games=[game_input],
        state=state,
        server_uri=server_uri,
    )

    proc_duration_ms = (time.perf_counter() - t_req_start) * 1000.0

    # Persistance SQL atomique stricte : confirmation HTTP 200 conditionnée au COMMIT
    reg_id = state.reg_info.get("model_id")
    cls_id = state.cls_info.get("model_id")
    if not reg_id or not cls_id:
        raise ServiceNotReadyError("Identifiants de modèles SQL non résolus au démarrage.")

    campaign_id = request.headers.get("X-Campaign-ID", "manual")
    try:
        record_prediction_transaction(
            pool=db_pool,
            request_id=req_uuid,
            started_at=started_at,
            endpoint="/predict",
            item_count=1,
            processing_duration_ms=round(proc_duration_ms, 2),
            inference_duration_ms=round(inf_duration_ms, 2),
            worker_pid=os.getpid(),
            instance_id=UUID(state.instance_id),
            api_revision=state.manifest.get("application_revision", "unknown"),
            campaign_id=campaign_id,
            results=results,
            games=[game_input],
            reg_model_id=reg_id,
            cls_model_id=cls_id,
        )
    except Exception as exc:
        logger.error(f"[PREDICT] Échec de persistance PostgreSQL : {exc}")
        record_api_request_error(
            pool=db_pool,
            request_id=req_uuid,
            started_at=started_at,
            endpoint="/predict",
            http_status=503,
            error_code="DEPENDENCY_UNAVAILABLE",
            processing_duration_ms=round(proc_duration_ms, 2),
            worker_pid=os.getpid(),
            instance_id=UUID(state.instance_id),
            api_revision=state.manifest.get("application_revision", "unknown"),
            campaign_id=campaign_id,
            item_count=1,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Erreur de persistance SQL : la transaction a été annulée ({exc}).",
        )

    return PredictionResponse(
        request_id=request_id,
        manifest_id=state.manifest_id,
        item_count=1,
        results=results,
        processing_duration_ms=round(proc_duration_ms, 2),
        inference_duration_ms=round(inf_duration_ms, 2),
    )


@app.post(
    "/predict/batch",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inférence"],
    responses={
        422: {"model": ErrorResponse, "description": "Validation de lot rejetée (1 à 100 jeux requis)"},
        503: {"model": ErrorResponse, "description": "Modèles ou base de données indisponibles"},
    },
)
def predict_batch(
    request: Request,
    payload: BatchPredictionRequest = Body(...),
) -> PredictionResponse:
    """Inférence conjointe vectorisée par lot avec persistance atomique de 2N prédictions."""
    t_req_start = time.perf_counter()
    request_id, req_uuid = get_request_ids(request)
    started_at = getattr(request.state, "started_at", datetime.now(timezone.utc))
    state: ModelServiceState = getattr(request.app.state, "service_state", None)

    if state is None or not state.is_ready:
        reason = state.ready_error if state else "Service non prêt"
        raise ServiceNotReadyError(f"Le service d'inférence n'est pas prêt : {reason}")

    games = payload.get_games()
    server_uri = state.manifest.get("mlflow_tracking_uri", settings.MLFLOW_TRACKING_URI)

    results, inf_duration_ms = run_prediction_batch(
        games=games,
        state=state,
        server_uri=server_uri,
    )

    proc_duration_ms = (time.perf_counter() - t_req_start) * 1000.0

    reg_id = state.reg_info.get("model_id")
    cls_id = state.cls_info.get("model_id")
    if not reg_id or not cls_id:
        raise ServiceNotReadyError("Identifiants de modèles SQL non résolus au démarrage.")

    campaign_id = request.headers.get("X-Campaign-ID", "manual")
    try:
        record_prediction_transaction(
            pool=db_pool,
            request_id=req_uuid,
            started_at=started_at,
            endpoint="/predict/batch",
            item_count=len(games),
            processing_duration_ms=round(proc_duration_ms, 2),
            inference_duration_ms=round(inf_duration_ms, 2),
            worker_pid=os.getpid(),
            instance_id=UUID(state.instance_id),
            api_revision=state.manifest.get("application_revision", "unknown"),
            campaign_id=campaign_id,
            results=results,
            games=games,
            reg_model_id=reg_id,
            cls_model_id=cls_id,
        )
    except Exception as exc:
        logger.error(f"[PREDICT BATCH] Échec de persistance PostgreSQL : {exc}")
        record_api_request_error(
            pool=db_pool,
            request_id=req_uuid,
            started_at=started_at,
            endpoint="/predict/batch",
            http_status=503,
            error_code="DEPENDENCY_UNAVAILABLE",
            processing_duration_ms=round(proc_duration_ms, 2),
            worker_pid=os.getpid(),
            instance_id=UUID(state.instance_id),
            api_revision=state.manifest.get("application_revision", "unknown"),
            campaign_id=campaign_id,
            item_count=len(games),
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Erreur de persistance SQL : transaction de lot annulée ({exc}).",
        )

    return PredictionResponse(
        request_id=request_id,
        manifest_id=state.manifest_id,
        item_count=len(games),
        results=results,
        processing_duration_ms=round(proc_duration_ms, 2),
        inference_duration_ms=round(inf_duration_ms, 2),
    )


@app.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Feedbacks"],
    responses={
        200: {"model": FeedbackResponse, "description": "Rejeu idempotent (même feedback_id et contenu)"},
        201: {"model": FeedbackResponse, "description": "Feedback créé avec succès"},
        404: {"model": ErrorResponse, "description": "Prédiction introuvable"},
        409: {"model": ErrorResponse, "description": "Conflit d'identifiant ou prédiction déjà annotée"},
        422: {"model": ErrorResponse, "description": "Incompatibilité de tâche ou validation en échec"},
        503: {"model": ErrorResponse, "description": "Base PostgreSQL indisponible"},
    },
)
def submit_feedback(
    request: Request,
    response: Response,
    payload: FeedbackInput = Body(...),
) -> Union[FeedbackResponse, JSONResponse]:
    """Réception et enregistrement transactionnel idempotent des retours d'utilisation."""
    request_id, req_uuid = get_request_ids(request)
    started_at = getattr(request.state, "started_at", datetime.now(timezone.utc))
    state: ModelServiceState = getattr(request.app.state, "service_state", None)

    if state is None or not state.is_ready:
        reason = state.ready_error if state else "Service non prêt"
        raise ServiceNotReadyError(f"Le service d'inférence n'est pas prêt : {reason}")

    campaign_id = request.headers.get("X-Campaign-ID", "manual")
    try:
        http_code, result = process_feedback_transaction(
            pool=db_pool,
            request_id=req_uuid,
            started_at=started_at,
            worker_pid=os.getpid(),
            instance_id=UUID(state.instance_id),
            api_revision=state.manifest.get("application_revision", "unknown"),
            campaign_id=campaign_id,
            feedback=payload,
        )
    except Exception as exc:
        logger.error(f"[FEEDBACK] Erreur inattendue : {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Erreur de persistance du feedback : {exc}",
        )

    response.status_code = http_code
    if http_code in (404, 409, 422):
        return JSONResponse(
            status_code=http_code,
            content=result,
            headers={"X-Request-ID": request_id},
        )

    return FeedbackResponse(**result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_prediction.main:app", host="127.0.0.1", port=8001, reload=False)
