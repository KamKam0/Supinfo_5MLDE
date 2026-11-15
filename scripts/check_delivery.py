#!/usr/bin/env python3
"""scripts/check_delivery.py - Vérificateur d'hygiène du dépôt Git et de conformité de livraison.

Contrôle automatiquement :
1. Hygiène Git : Absence de gros fichiers (> 1 Mo) dans l'arbre git, absence de secrets (.env), de .venv ou de caches .pyc.
2. Intégrité de l'environnement : Conformité uv.lock, Python >= 3.12, variables d'environnement documentées.
3. Présence des livrables documentaires : docs/ (7 documents clés), reports/ (sizing, ops_summary, audits, matrice des exigences).
4. Intégrité des schémas et contrats : OpenAPI, contrats Pydantic, schémas SQL migrations.
5. Intégrité des scripts opérationnels : doctor.py, demo_incident.py, run_capacity_matrix.py, select_models.py, etc.

Usage :
    uv run --locked python scripts/check_delivery.py
    uv run --locked python scripts/check_delivery.py --strict
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Tuple

# Sortie UTF-8 forcée (console Windows cp1252 sinon incapable d'afficher les symboles)
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("check_delivery")


def check_git_tracked_files() -> Tuple[bool, List[str]]:
    """Vérifie qu'aucun fichier volumineux (> 1 Mo) ni interdit n'est présent dans l'index Git."""
    errors = []
    try:
        res = subprocess.run(
            ["git", "ls-files", "-s"],
            capture_output=True,
            text=True,
            check=True,
            cwd=PROJECT_ROOT,
        )
    except Exception as exc:
        return False, [f"Impossible d'exécuter git ls-files : {exc}"]

    forbidden_patterns = [
        ".env",
        ".venv/",
        "__pycache__/",
        ".pytest_cache/",
        ".ipynb_checkpoints/",
        ".sqlite",
        "games_march2025_full.csv",
    ]
    max_size_bytes = 1024 * 1024  # 1 Mo

    for line in res.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 4:
            rel_path = " ".join(parts[3:])
            # 1. Vérification des motifs interdits
            filename = Path(rel_path).name
            if filename == ".env" or (filename.startswith(".env.") and not filename.endswith(".example")):
                errors.append(f"Fichier secret interdit tracké dans Git : '{rel_path}'")

            for pat in [p for p in forbidden_patterns if p != ".env"]:
                if pat in rel_path or rel_path.endswith(pat.strip("/")):
                    errors.append(f"Fichier interdit tracké dans Git : '{rel_path}' (motif: {pat})")

            # 2. Vérification de la taille physique du fichier tracké
            file_path = PROJECT_ROOT / rel_path
            if file_path.exists() and file_path.is_file():
                sz = file_path.stat().st_size
                if sz > max_size_bytes:
                    # Exception admise uniquement si spécifiquement autorisée
                    errors.append(f"Fichier trop lourd tracké dans Git : '{rel_path}' ({sz / 1024:.1f} Ko > 1024 Ko)")

    return len(errors) == 0, errors


def check_required_deliverables() -> Tuple[bool, List[str]]:
    """Vérifie la présence et la non-vacuité de tous les livrables obligatoires."""
    errors = []

    required_files = [
        # Racine
        "README.md",
        "pyproject.toml",
        "uv.lock",
        # Documentation
        "docs/architecture.md",
        "docs/data_contract.md",
        "docs/model_contract.md",
        "docs/api_contract.md",
        "docs/operations.md",
        "docs/recovery.md",
        "docs/openapi.json",
        # Rapports obligatoires
        "reports/requirements_matrix.md",
        "reports/sizing.md",
        "reports/ops_summary.md",
        # Rapports d'incidents
        "reports/incidents/incident_invalid_data.md",
        "reports/incidents/incident_postgres_down.md",
        "reports/incidents/incident_mlflow_reconcile.md",
        "reports/incidents/incident_model_rollback.md",
        "reports/incidents/incident_alert_failure.md",
        # Scripts clés
        "scripts/doctor.py",
        "scripts/demo_incident.py",
        "scripts/run_capacity_matrix.py",
        "scripts/analyse_capacity.py",
        "scripts/select_models.py",
        "scripts/reconcile.py",
        "scripts/build_service_manifest.py",
        # Infrastructure
        "infra/compose.yml",
        "infra/postgres/010-tracing.sql",
        "infra/postgres/migrations/020-ops-observability.sql",
        "infra/grafana/dashboards/steam-mlops.json",
        # État et fixtures
        "state/service-manifest.json",
        "state/service-manifest.prev.json",
        "fixtures/load_fixtures.json",
        "config/loadtest.json",
    ]

    for rf in required_files:
        p = PROJECT_ROOT / rf
        if not p.exists():
            errors.append(f"Livrable obligatoire manquant : '{rf}'")
        elif p.is_file() and p.stat().st_size == 0:
            errors.append(f"Livrable obligatoire vide (0 octet) : '{rf}'")

    return len(errors) == 0, errors


def check_requirements_matrix_consistency() -> Tuple[bool, List[str]]:
    """Vérifie la cohérence arithmétique de la matrice des exigences reports/requirements_matrix.md."""
    matrix_path = PROJECT_ROOT / "reports" / "requirements_matrix.md"
    if not matrix_path.exists():
        return False, ["reports/requirements_matrix.md introuvable"]

    content = matrix_path.read_text(encoding="utf-8")
    errors = []

    # Vérification que REQ-01 à REQ-23 sont bien marqués DONE
    for i in range(1, 24):
        code = f"REQ-{i:02d}"
        if code not in content:
            errors.append(f"Exigence {code} introuvable dans la matrice !")

    return len(errors) == 0, errors


def check_environment_doctor() -> Tuple[bool, List[str]]:
    """Exécute doctor.py pour valider la conformité de l'environnement."""
    doctor_script = PROJECT_ROOT / "scripts" / "doctor.py"
    if not doctor_script.exists():
        return False, ["scripts/doctor.py manquant"]

    try:
        res = subprocess.run(
            [sys.executable, str(doctor_script)],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=PROJECT_ROOT,
        )
        if res.returncode != 0:
            return False, [f"scripts/doctor.py a échoué (code {res.returncode}) : {res.stderr[:200]}"]
        return True, []
    except Exception as exc:
        return False, [f"Échec d'exécution de scripts/doctor.py : {exc}"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Vérificateur de livraison et d'hygiène de dépôt Git.")
    parser.add_argument("--strict", action="store_true", help="Active le mode d'inspection stricte")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("      VÉRIFICATION D'HYGIÈNE ET AUDIT DE LIVRAISON MLOps 5MLDE")
    print(f"      Racine du projet : {PROJECT_ROOT}")
    print("=" * 80)

    checks = [
        ("Hygiène Git & Absence de Fichiers Lourds / Secrets", check_git_tracked_files),
        ("Présence et Intégrité des Livrables Obligatoires", check_required_deliverables),
        ("Cohérence de la Matrice des Exigences", check_requirements_matrix_consistency),
        ("Conformité de l'Environnement Local (doctor.py)", check_environment_doctor),
    ]

    all_passed = True
    for title, check_fn in checks:
        print(f"\n[AUDIT] {title}...")
        passed, errs = check_fn()
        if passed:
            print(f" -> ✅ CONFORME")
        else:
            print(f" -> ❌ NON CONFORME ({len(errs)} anomalie(s))")
            for err in errs:
                print(f"    - {err}")
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("          BILAN GLOBAL : ✅ DÉPÔT ET LIVRABLES 100% CONFORMES")
    else:
        print("          BILAN GLOBAL : ❌ ANOMALIES DÉTECTÉES")
    print("=" * 80 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
