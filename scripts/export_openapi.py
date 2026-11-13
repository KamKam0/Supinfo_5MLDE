"""scripts/export_openapi.py - Export programmatique du schéma OpenAPI.

Génère la spécification OpenAPI JSON officielle de l'API d'inférence FastAPI
vers `docs/openapi.json` sans écriture manuelle.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

# Assurer l'accès au module racine
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from api_prediction.main import app

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("export_openapi")

OUTPUT_PATH = project_root / "docs" / "openapi.json"


def export_openapi(output_path: Path = OUTPUT_PATH) -> Path:
    """Extrait le schéma OpenAPI de l'application FastAPI et le persiste sous format JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    openapi_schema = app.openapi()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(openapi_schema, f, indent=2, ensure_ascii=False)

    paths_count = len(openapi_schema.get("paths", {}))
    schemas_count = len(openapi_schema.get("components", {}).get("schemas", {}))
    logger.info(
        f"[OPENAPI] Schéma OpenAPI exporté avec succès vers {output_path} "
        f"({paths_count} routes, {schemas_count} schémas définis)."
    )
    return output_path


if __name__ == "__main__":
    export_openapi()
