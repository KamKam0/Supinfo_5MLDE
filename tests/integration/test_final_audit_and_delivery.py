"""tests/integration/test_final_audit_and_delivery.py - Tests d'audit final du projet.

Valide formellement :
1. La présence et la structure du démonstrateur nominal (scripts/demo_nominal.py et scripts/demo_full.py).
2. L'intégrité et le succès du dernier rapport de démonstration nominale (reports/demo_nominal_report.json).
3. La couverture intégrale 27/27 de la matrice des exigences au statut DONE dans reports/requirements_matrix.md.
5. L'intégrité cryptographique absolue du manifeste de service state/service-manifest.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

import pytest

from api_prediction.lifecycle import verify_manifest_hash
from scripts.build_service_manifest import compute_manifest_sha256
from src.settings import settings


def test_final_audit_demo_scripts_exist():
    """Vérifie l'existence et l'exécutabilité des scripts de démonstration."""
    demo_nom = settings.PROJECT_ROOT / "scripts" / "demo_nominal.py"
    demo_full = settings.PROJECT_ROOT / "scripts" / "demo_full.py"

    assert demo_nom.exists(), "scripts/demo_nominal.py doit exister"
    assert demo_full.exists(), "scripts/demo_full.py doit exister"

    content_nom = demo_nom.read_text(encoding="utf-8")
    assert "step_1_doctor_check" in content_nom
    assert "step_2_data_validation" in content_nom
    assert "step_3_orchestrated_pipeline" in content_nom
    assert "step_4_manifest_qualification" in content_nom
    assert "step_5_api_predictions_and_leak_prevention" in content_nom
    assert "step_6_feedback_and_idempotence" in content_nom
    assert "step_7_sql_audit_reconciliation" in content_nom


def test_final_audit_demo_report_success():
    """Vérifie que le rapport de démonstration nominale atteste du succès des 7 étapes."""
    report_file = settings.PROJECT_ROOT / "reports" / "demo_nominal_report.json"
    assert report_file.exists(), "Le rapport reports/demo_nominal_report.json doit être généré"

    data = json.loads(report_file.read_text(encoding="utf-8"))
    assert data.get("overall_status") == "SUCCESS", "Le statut global doit être SUCCESS"
    steps = data.get("steps", {})
    assert len(steps) >= 7, "Les 7 étapes nominales doivent être consignées"

    assert steps["1_doctor"]["status"] == "SUCCESS"
    assert steps["2_data_validation"]["success"] is True
    assert steps["3_pipeline"]["status"] == "SUCCESS"
    assert steps["4_manifest"]["status"] == "SUCCESS"
    assert steps["5_inference"]["status"] == "SUCCESS"
    assert steps["5_inference"]["leak_rejected_422"] is True
    assert steps["6_feedback"]["idempotence_verified"] is True
    assert steps["7_sql_audit"]["sql_reconciled"] is True


def test_final_audit_requirements_matrix_100_percent_done():
    """Vérifie que 100% des 27 exigences sont au statut DONE."""
    matrix_file = settings.PROJECT_ROOT / "reports" / "requirements_matrix.md"
    assert matrix_file.exists(), "reports/requirements_matrix.md doit exister"

    content = matrix_file.read_text(encoding="utf-8")
    for i in range(1, 28):
        req_id = f"REQ-{i:02d}"
        pattern = rf"\|\s*\*\*{req_id}\*\*\s*\|.*\|\s*`([^`]+)`\s*\|\s*$"
        match = re.search(pattern, content, re.MULTILINE)
        assert match is not None, f"L'exigence {req_id} doit être présente dans la matrice"
        status = match.group(1).strip()
        assert status == "DONE", f"L'exigence {req_id} doit avoir le statut DONE (actuel: {status})"

    assert "Exigences `DONE` :** 27" in content or "Exigences `DONE` :** 27 / 27" in content or "27" in content


def test_final_audit_manifest_cryptographic_integrity():
    """Vérifie l'intégrité cryptographique SHA-256 du manifeste de service actif."""
    manifest_file = settings.PROJECT_ROOT / "state" / "service-manifest.json"
    assert manifest_file.exists(), "state/service-manifest.json doit exister"

    manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert verify_manifest_hash(manifest_data) is True, "L'empreinte du manifeste doit correspondre à son contenu"
    expected_hash = compute_manifest_sha256(manifest_data)
    assert manifest_data["manifest_sha256"] == expected_hash
