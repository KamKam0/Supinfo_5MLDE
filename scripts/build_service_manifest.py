"""scripts/build_service_manifest.py - Générateur du manifeste de service immuable.

Construit un manifeste formel figeant les identités exactes des versions de modèles
à instancier en production (sans résolution dynamique d'alias au runtime d'inférence).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Optional
from uuid import uuid4

# Assurer l'accès au module racine src
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.settings import settings
from src.tracking.registry import get_client, get_model_version_by_alias

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_MANIFEST = project_root / "state" / "service-manifest.json"
REGRESSION_MODEL_NAME = "steam-positive-ratio"
CLASSIFICATION_MODEL_NAME = "steam-owner-range"


def get_git_commit_sha() -> str:
    """Retourne l'empreinte SHA du commit Git courant ou 'unknown'."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown"


def compute_manifest_sha256(manifest_data: Dict[str, Any]) -> str:
    """Calcule l'empreinte SHA-256 canonique du manifeste en excluant la clé de hachage."""
    copy_data = {k: v for k, v in manifest_data.items() if k != "manifest_sha256"}
    canonical_json = json.dumps(copy_data, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest().lower()


def build_service_manifest(
    alias: str = "champion",
    selection_path: Optional[Path | str] = None,
    output_path: Optional[Path | str] = None,
    mode: str = "qualified",
    tracking_uri: Optional[str] = None,
) -> Dict[str, Any]:
    """Construit et publie de manière atomique le manifeste de service actif."""
    out_file = Path(output_path or DEFAULT_OUTPUT_MANIFEST).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    uri = tracking_uri or settings.MLFLOW_TRACKING_URI
    client = get_client(tracking_uri=uri)

    manifest_id = f"man-{uuid4().hex[:8]}"
    now_iso = datetime.now(timezone.utc).isoformat()

    # 1. Résolution des versions exactes à partir de l'alias ou de la sélection
    reg_ver: Optional[str] = None
    cls_ver: Optional[str] = None
    reg_run_id: Optional[str] = None
    cls_run_id: Optional[str] = None
    selection_summary: Dict[str, Any] = {}

    if selection_path and Path(selection_path).exists():
        with open(selection_path, "r", encoding="utf-8") as f:
            selection_summary = json.load(f)
        candidate_run = selection_summary.get("candidate_run_id")
        verdict = selection_summary.get("verdict", "UNKNOWN")

        if verdict == "DEMO_ONLY" and mode == "qualified":
            raise ValueError(
                "[MANIFEST] Violation de gouvernance : interdiction absolue d'émettre un manifeste "
                "'qualified' à partir d'une sélection étiquetée DEMO_ONLY."
            )

    # Résolution via le Model Registry MLflow
    clean_alias = alias.lstrip("@")
    reg_ver = get_model_version_by_alias(REGRESSION_MODEL_NAME, clean_alias, client=client)
    cls_ver = get_model_version_by_alias(CLASSIFICATION_MODEL_NAME, clean_alias, client=client)

    if not reg_ver:
        raise ValueError(f"Modèle '{REGRESSION_MODEL_NAME}' introuvable avec l'alias '@{clean_alias}'.")
    if not cls_ver:
        raise ValueError(f"Modèle '{CLASSIFICATION_MODEL_NAME}' introuvable avec l'alias '@{clean_alias}'.")

    # Récupération des métadonnées détaillées des versions
    mv_reg = client.get_model_version(REGRESSION_MODEL_NAME, str(reg_ver))
    mv_cls = client.get_model_version(CLASSIFICATION_MODEL_NAME, str(cls_ver))

    reg_run_id = mv_reg.run_id
    cls_run_id = mv_cls.run_id

    # Récupération de l'ID du manifeste précédent si existant
    prev_manifest_id: Optional[str] = None
    if out_file.exists():
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                old_data = json.load(f)
                prev_manifest_id = old_data.get("manifest_id")
        except Exception:
            pass

    manifest_data: Dict[str, Any] = {
        "manifest_id": manifest_id,
        "schema_version": "1.0.0",
        "generated_at": now_iso,
        "mode": mode,
        "application_revision": get_git_commit_sha(),
        "input_contract_version": "1.0.0",
        "mlflow_tracking_uri": uri,
        "models": {
            "regression": {
                "model_name": REGRESSION_MODEL_NAME,
                "version": str(reg_ver),
                "model_uri": f"models:/{REGRESSION_MODEL_NAME}/{reg_ver}",
                "mlflow_run_id": reg_run_id,
                "target_column": settings.TARGET_REGRESSION,
                "unit": "pct_pos_total [0, 100]",
            },
            "classification": {
                "model_name": CLASSIFICATION_MODEL_NAME,
                "version": str(cls_ver),
                "model_uri": f"models:/{CLASSIFICATION_MODEL_NAME}/{cls_ver}",
                "mlflow_run_id": cls_run_id,
                "target_column": settings.TARGET_CLASSIFICATION,
                "classes_count": 12,
            },
        },
        "target_alias": clean_alias,
        "previous_manifest_id": prev_manifest_id,
        "selection_metadata": selection_summary,
    }

    # Calcul de l'empreinte SHA-256
    manifest_data["manifest_sha256"] = compute_manifest_sha256(manifest_data)

    # Sauvegarde du manifeste précédent avant écrasement
    if out_file.exists():
        prev_file = out_file.with_name(out_file.stem + ".prev" + out_file.suffix)
        try:
            import shutil
            shutil.copy2(out_file, prev_file)
            logger.info(f"[MANIFEST] Ancien manifeste sauvegardé vers {prev_file.name}")
        except Exception as exc:
            logger.warning(f"[MANIFEST] Impossible de sauvegarder l'ancien manifeste dans {prev_file} : {exc}")

    # Écriture atomique
    tmp_out = out_file.with_suffix(".tmp")
    with open(tmp_out, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)
    tmp_out.replace(out_file)

    logger.info(
        f"[MANIFEST] Manifeste de service généré avec succès : {out_file.name} "
        f"(id: {manifest_id}, Reg: v{reg_ver}, Cls: v{cls_ver}, SHA256: {manifest_data['manifest_sha256'][:12]}...)"
    )
    return manifest_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Générateur du manifeste de service d'inférence")
    parser.add_argument("--alias", type=str, default="champion", help="Alias cible dans MLflow Registry")
    parser.add_argument("--selection", type=str, default=None, help="Chemin vers le fichier de sélection approuvée")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_MANIFEST), help="Chemin du manifeste de sortie")
    parser.add_argument("--mode", type=str, default="qualified", choices=["qualified", "demo"], help="Mode de publication")
    args = parser.parse_args()

    manifest = build_service_manifest(
        alias=args.alias,
        selection_path=args.selection,
        output_path=args.output,
        mode=args.mode,
    )
    print("\n--- MANIFESTE DE SERVICE GÉNÉRÉ ---")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
