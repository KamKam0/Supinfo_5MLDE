"""api_trigger_passif/main.py - Service FastAPI de collecte passive.

Expose sur le port 8000 :
- POST /games : Réception transactionnelle d'événements labellisés avec détection automatique du seuil de 500 jeux.
- GET /status : Consultation en temps réel des compteurs et lots actifs.
- GET /batches/{batch_id} : Détails d'un lot durable et traçabilité.
- GET /health : Sonde de santé du service.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from api_trigger_passif.storage import PushStorage
from api_trigger_passif.validation import PushEventPayload


def create_app(db_path: Optional[Path | str] = None) -> FastAPI:
    """Factory de création de l'application FastAPI de collecte passive."""
    app = FastAPI(
        title="Steam MLOps - Collecte Passive PUSH & Déclencheur par Lot",
        description="Réception passive d'observations Steam labellisées sur le port 8000.",
        version="1.0.0",
    )

    storage = PushStorage(db_path=db_path)
    app.state.storage = storage

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        safe_errors = [
            {"loc": list(err.get("loc", ())), "msg": str(err.get("msg", "")), "type": str(err.get("type", ""))}
            for err in exc.errors()
        ]
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "Unprocessable Entity",
                "message": "Payload JSON invalide ou données non conformes aux contrats.",
                "details": safe_errors,
            },
        )

    @app.get("/health", tags=["Supervision"])
    def health_check() -> Dict[str, Any]:
        """Sonde de santé de l'API passive."""
        return {
            "status": "healthy",
            "service": "api_trigger_passif",
            "port": 8000,
        }

    @app.get("/status", tags=["Supervision"])
    def get_collector_status() -> Dict[str, Any]:
        """Retourne les compteurs de réception et la liste des lots."""
        return storage.get_status()

    @app.get("/batches/{batch_id}", tags=["Lots"])
    def get_batch_details(batch_id: str) -> Dict[str, Any]:
        """Retourne le statut d'un lot et la liste des jeux associés."""
        batch = storage.get_batch(batch_id)
        if not batch:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Lot '{batch_id}' introuvable.",
            )
        return batch

    @app.post("/games", tags=["Collecte"])
    async def receive_game_event(request: Request, response: Response) -> Dict[str, Any]:
        """Réceptionne une observation labellisée.

        Réponses :
        - 202 Accepted : Nouvelle observation validée. Si un seuil de 500 jeux non affectés est atteint,
                         un lot est immédiatement créé dans la même transaction.
        - 200 OK : Événement strictement identique déjà reçu (idempotence).
        - 409 Conflict : Même event_id mais contenu différent.
        - 422 Unprocessable Entity : Données manquantes ou schéma non conforme.
        """
        raw_body = await request.body()
        if not raw_body:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Corps de requête vide.",
            )

        try:
            event = PushEventPayload.model_validate_json(raw_body)
        except ValidationError as exc:
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "error": "Unprocessable Entity",
                    "message": "Erreur de validation du schéma PUSH.",
                    "details": json.loads(exc.json()),
                },
            )

        http_code, resp_data = storage.record_event(event, raw_body)
        response.status_code = http_code
        return resp_data

    return app


app = create_app()
