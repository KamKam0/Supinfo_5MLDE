"""api_prediction/main.py - Application FastAPI d'Inférence Steam (Port 8001).

Expose le service d'inférence en ligne haute performance :
- POST /predict : Inférence conjointe unitaire.
- POST /predict/batch : Inférence conjointe vectorisée par lot (1 à 100 jeux).
- GET /health/live : Sonde de vivacité élémentaire (HTTP 200).
- GET /health/ready : Sonde d'aptitude opérationnelle (HTTP 200 ou 503).
- GET /models : Consultation du manifeste actif et des versions de modèles figées.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import time
from typing import Any, Union
from uuid import uuid4

from fastapi import Body, FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api_prediction.lifecycle import ModelServiceState, lifespan
from api_prediction.schemas import (
    BatchPredictionRequest,
    ErrorResponse,
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


@app.middleware("http")
async def correlation_and_headers_middleware(request: Request, call_next: Any) -> Response:
    """Injecte un identifiant de corrélation universel et les en-têtes d'audit de versions."""
    request_id = request.headers.get("X-Request-ID") or f"req-{uuid4().hex[:12]}"
    request.state.request_id = request_id

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
    """Gestionnaire d'erreurs de validation 422 avec formatage standard sans fuite."""
    request_id = getattr(request.state, "request_id", f"req-{uuid4().hex[:12]}")
    error_payload = ErrorResponse(
        request_id=request_id,
        error_code="INVALID_INPUT",
        message="Erreur de validation de la requête d'inférence (contrat strict sans variables cibles).",
        details=exc.errors(),
    ).model_dump()

    return JSONResponse(
        status_code=422,
        content=error_payload,
        headers={"X-Request-ID": request_id},
    )


@app.exception_handler(ServiceNotReadyError)
async def service_not_ready_handler(request: Request, exc: ServiceNotReadyError) -> JSONResponse:
    """Gestionnaire d'indisponibilité du service 503."""
    request_id = getattr(request.state, "request_id", f"req-{uuid4().hex[:12]}")
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
    request_id = getattr(request.state, "request_id", f"req-{uuid4().hex[:12]}")
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
    """Sonde d'aptitude opérationnelle (retourne HTTP 503 si les modèles ne sont pas prêts)."""
    state: ModelServiceState = getattr(request.app.state, "service_state", None)
    now_iso = datetime.now(timezone.utc).isoformat()

    if state is None or not state.is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(
            status="not_ready",
            service="steam-prediction-api",
            instance_id=state.instance_id if state else None,
            manifest_id=state.manifest_id if state else None,
            models_loaded=False,
            timestamp=now_iso,
            error=state.ready_error if state else "Worker non initialisé",
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
        503: {"model": ErrorResponse, "description": "Modèles ou dépendance indisponibles"},
    },
)
def predict_single(
    request: Request,
    payload: Union[SinglePredictionRequest, InferenceGameInput] = Body(...),
) -> PredictionResponse:
    """Inférence unitaire conjointe sur un jeu Steam unique."""
    t_req_start = time.perf_counter()
    request_id = getattr(request.state, "request_id", f"req-{uuid4().hex[:12]}")
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
        503: {"model": ErrorResponse, "description": "Modèles ou dépendance indisponibles"},
    },
)
def predict_batch(
    request: Request,
    payload: BatchPredictionRequest = Body(...),
) -> PredictionResponse:
    """Inférence conjointe vectorisée sur un lot de 1 à 100 jeux Steam."""
    t_req_start = time.perf_counter()
    request_id = getattr(request.state, "request_id", f"req-{uuid4().hex[:12]}")
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

    return PredictionResponse(
        request_id=request_id,
        manifest_id=state.manifest_id,
        item_count=len(games),
        results=results,
        processing_duration_ms=round(proc_duration_ms, 2),
        inference_duration_ms=round(inf_duration_ms, 2),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_prediction.main:app", host="127.0.0.1", port=8001, reload=False)
