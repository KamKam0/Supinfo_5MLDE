"""tests/integration/test_delivery_and_reproducibility.py - Tests de reproductibilité et conformité du dépôt.

Valide :
- L'audit complet d'hygiène Git et de conformité de livraison via scripts/check_delivery.py.
- L'absence de fichiers lourds (> 1 Mo), secrets ou caches dans l'index Git.
- La complétude intégrale du corpus documentaire (docs/) et des rapports d'audit (reports/).
- La validité de la matrice des exigences et la non-régression de l'environnement.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import pytest

from scripts.check_delivery import (
    check_environment_doctor,
    check_git_tracked_files,
    check_required_deliverables,
    check_requirements_matrix_consistency,
)
from src.settings import settings


def test_delivery_git_hygiene_and_no_secrets():
    """Vérifie l'absence absolue de secrets, de gros fichiers (> 1 Mo) ou de caches dans Git."""
    passed, errors = check_git_tracked_files()
    assert passed is True, f"Anomalies d'hygiène Git détectées : {errors}"


def test_delivery_required_documents_and_reports_exist():
    """Vérifie que l'ensemble des 8 documents docs/ et des rapports reports/ sont présents et non vides."""
    passed, errors = check_required_deliverables()
    assert passed is True, f"Livrables obligatoires manquants ou vides : {errors}"


def test_delivery_requirements_matrix_integrity():
    """Vérifie que la matrice des exigences contient bien l'ensemble des exigences-01 à REQ-23."""
    passed, errors = check_requirements_matrix_consistency()
    assert passed is True, f"Incohérence dans la matrice des exigences : {errors}"


def test_delivery_environment_doctor_is_healthy():
    """Vérifie que le script d'infrastructure doctor.py valide l'ensemble des composants."""
    passed, errors = check_environment_doctor()
    assert passed is True, f"Diagnostic doctor.py en échec : {errors}"


def test_delivery_cli_script_execution():
    """Exécute directement scripts/check_delivery.py en sous-processus et vérifie le code retour 0."""
    script_path = settings.PROJECT_ROOT / "scripts" / "check_delivery.py"
    res = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=settings.PROJECT_ROOT,
        timeout=30,
    )
    assert res.returncode == 0
    assert "BILAN GLOBAL : ✅ DÉPÔT ET LIVRABLES 100% CONFORMES" in res.stdout
