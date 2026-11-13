"""api_prediction/telemetry.py - Télémétrie de secours et journalisation locale.

En cas d'indisponibilité de PostgreSQL ou d'échec critique de persistance :
- Consigne les traces d'erreurs dans un journal local JSON Lines immuable (`state/telemetry_fallback.jsonl`).
- Évite tout double enregistrement entre les middlewares et les handlers d'exception.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from src.settings import settings

logger = logging.getLogger(__name__)

FALLBACK_LOG_PATH = settings.PROJECT_ROOT / "state" / "telemetry_fallback.jsonl"


def log_telemetry_fallback(
    request_id: str,
    endpoint: str,
    http_status: int,
    error_code: str,
    message: str,
    details: Optional[Any] = None,
    output_path: Path = FALLBACK_LOG_PATH,
) -> None:
    """Écrit de manière append-only un événement d'incident dans le journal local de secours."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    event: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": str(request_id),
        "endpoint": endpoint,
        "http_status": http_status,
        "error_code": error_code,
        "message": message,
        "details": details,
    }
    try:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        logger.warning(
            f"[TELEMETRY FALLBACK] Trace d'incident enregistrée localement pour {request_id} ({error_code})."
        )
    except Exception as exc:
        logger.error(f"[TELEMETRY CRITICAL] Impossible d'écrire dans {output_path} : {exc}")
