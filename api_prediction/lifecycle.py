"""api_prediction/lifecycle.py - Gestionnaire de cycle de vie FastAPI.

Assure les prérequis stricts de démarrage d'un worker Uvicorn :
1. Lecture et validation cryptographique du manifeste de service actif (SHA-256).
2. Génération d'un UUID unique `instance_id` pour le cycle de vie du worker.
3. Téléchargement et instanciation des deux pipelines en mémoire UNE SEULE FOIS.
4. Exécution d'une sonde de pré-chauffe (*warm-up*) interne isolée.
5. Déclaration de l'état `ready` uniquement après validation complète de ces étapes.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional
from uuid import uuid4
from fastapi import FastAPI
import pandas as pd

from src.settings import settings
from src.tracking.registry import load_model

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST_PATH = settings.PROJECT_ROOT / "state" / "service-manifest.json"


def verify_manifest_hash(manifest_data: Dict[str, Any]) -> bool:
    """Vérifie l'intégrité cryptographique SHA-256 du manifeste de service."""
    claimed_hash = manifest_data.get("manifest_sha256")
    if not claimed_hash:
        return False
    copy_data = {k: v for k, v in manifest_data.items() if k != "manifest_sha256"}
    canonical_json = json.dumps(copy_data, sort_keys=True, ensure_ascii=True)
    computed_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest().lower()
    return claimed_hash.lower() == computed_hash


def run_warmup_probe(reg_model: Any, cls_model: Any) -> None:
    """Exécute une prédiction de warm-up interne en mémoire sans persistance."""
    synthetic_probe = pd.DataFrame([
        {
            "price": 14.99,
            "required_age": 0,
            "dlc_count": 1,
            "achievements": 25,
            "windows": True,
            "mac": False,
            "linux": False,
            "genres": ["Action", "Indie"],
            "release_date": "2023-01-01",
        }
    ])
    logger.info("[WARMUP] Exécution de la sonde de pré-chauffe interne...")
    reg_val = reg_model.predict(synthetic_probe)
    cls_val = cls_model.predict(synthetic_probe)

    if len(reg_val) != 1 or len(cls_val) != 1:
        raise RuntimeError("[WARMUP] Échec de la sonde : cardinalité de sortie anormale.")
    if not (0.0 <= float(reg_val[0]) <= 100.0):
        raise RuntimeError(f"[WARMUP] Sortie régression hors contrat [0, 100] : {reg_val[0]}")
    if not str(cls_val[0]).strip():
        raise RuntimeError("[WARMUP] Sortie classification vide.")

    logger.info(
        f"[WARMUP] Sonde réussie (reg: {float(reg_val[0]):.2f}%, cls: '{cls_val[0]}'). Worker chaud."
    )


@dataclass
class ModelServiceState:
    """État applicatif en mémoire du service d'inférence pour un worker."""

    manifest_path: Path
    manifest: Dict[str, Any]
    manifest_id: str
    manifest_sha256: str
    mode: str
    instance_id: str
    reg_model: Any = None
    cls_model: Any = None
    reg_info: Dict[str, Any] = field(default_factory=dict)
    cls_info: Dict[str, Any] = field(default_factory=dict)
    is_ready: bool = False
    ready_error: Optional[str] = None
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Gestionnaire de cycle de vie FastAPI garantissant l'initialisation unique."""
    manifest_path_str = os.environ.get("SERVICE_MANIFEST_PATH")
    manifest_path = Path(manifest_path_str).resolve() if manifest_path_str else DEFAULT_MANIFEST_PATH.resolve()

    instance_id = f"inst-{uuid4().hex[:8]}"
    logger.info(
        f"[LIFESPAN] Démarrage du worker {instance_id}. Manifeste cible : {manifest_path}"
    )

    state = ModelServiceState(
        manifest_path=manifest_path,
        manifest={},
        manifest_id="unknown",
        manifest_sha256="",
        mode="unknown",
        instance_id=instance_id,
        is_ready=False,
    )
    app.state.service_state = state

    try:
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifeste de service introuvable à l'emplacement : {manifest_path}"
            )

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        if not verify_manifest_hash(manifest_data):
            raise ValueError(
                f"Altération détectée : le hash SHA-256 du manifeste {manifest_path} est invalide."
            )

        state.manifest = manifest_data
        state.manifest_id = manifest_data.get("manifest_id", "man-unknown")
        state.manifest_sha256 = manifest_data.get("manifest_sha256", "")
        state.mode = manifest_data.get("mode", "qualified")

        models_meta = manifest_data.get("models", {})
        reg_meta = models_meta.get("regression", {})
        cls_meta = models_meta.get("classification", {})
        state.reg_info = reg_meta
        state.cls_info = cls_meta

        tracking_uri = manifest_data.get("mlflow_tracking_uri", settings.MLFLOW_TRACKING_URI)

        reg_uri = reg_meta.get("model_uri")
        cls_uri = cls_meta.get("model_uri")

        if not reg_uri or not cls_uri:
            raise ValueError(
                "Le manifeste est incomplet : URI absente pour la régression ou la classification."
            )

        # Chargement unique en mémoire des deux pipelines
        logger.info(f"[LIFESPAN] Chargement en mémoire du pipeline régression : {reg_uri}")
        reg_model = load_model(reg_uri, tracking_uri=tracking_uri)

        logger.info(f"[LIFESPAN] Chargement en mémoire du pipeline classification : {cls_uri}")
        cls_model = load_model(cls_uri, tracking_uri=tracking_uri)

        # Exécution de la sonde de warm-up
        run_warmup_probe(reg_model, cls_model)

        state.reg_model = reg_model
        state.cls_model = cls_model
        state.is_ready = True
        state.ready_error = None
        logger.info(
            f"[LIFESPAN] Worker {instance_id} prêt et opérationnel sur le manifeste {state.manifest_id}."
        )

    except Exception as exc:
        state.is_ready = False
        state.ready_error = str(exc)
        logger.error(f"[LIFESPAN] Échec d'initialisation du worker {instance_id} : {exc}")

    yield

    # Arrêt propre du worker
    logger.info(f"[LIFESPAN] Arrêt du worker {instance_id}.")
    state.is_ready = False
