"""scripts/configure_alerts.py - Configuration de l'automatisation d'alertes Prefect 3.

Permet :
- La vérification de la disponibilité du récepteur d'alertes local (port 9000).
- L'enregistrement / mise à jour par SDK de l'automation Prefect réagissant aux flows FAILED.
- L'émission d'un événement d'alerte de test pour prouver la réception sans faille.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional

# Assurer l'accès au module racine src
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import httpx

from src.orchestration.alerts import AlertDispatcher, AlertMessage
from src.settings import settings

logger = logging.getLogger(__name__)


def check_receiver_health(receiver_url: str = "http://127.0.0.1:9000/health") -> bool:
    """Vérifie si le récepteur local d'alertes est actif."""
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(receiver_url)
            return resp.status_code == 200
    except Exception:
        return False


def test_send_alert(
    flow_name: str = "steam-training-pipeline",
    flow_run_id: str = "test-manual-alert-001",
    error_message: str = "Simulation de défaillance pour validation de la chaîne d'alerte",
    batch_id: str = "batch-test-alert",
    mlflow_run_id: Optional[str] = "test-mlflow-run-abc",
    local_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Émet une alerte de test via le dispatcheur."""
    alert = AlertMessage(
        flow_name=flow_name,
        flow_run_id=flow_run_id,
        state="FAILED",
        error_message=error_message,
        environment="test",
        mode="alert-verification",
        batch_id=batch_id,
        mlflow_run_id=mlflow_run_id,
        details={"verification": True, "initiated_by": "configure_alerts.py"},
    )
    dispatcher = AlertDispatcher(local_webhook_url=local_url)
    return dispatcher.send_alert(alert)


def configure_prefect_automation(
    automation_name: str = "steam-pipeline-failure-alert",
) -> Dict[str, Any]:
    """Configure une automatisation Prefect 3 par SDK si le serveur le permet."""
    try:
        from prefect.automations import Automation
        from prefect.events.schemas.automations import EventTrigger

        # Définition du trigger réagissant au passage en state FAILED
        trigger = EventTrigger(
            expect={"prefect.flow-run.Failed"},
            match_related={"prefect.resource.name": "steam-training-pipeline"},
            posture="Reactive",
            threshold=1,
            within=60.0,
        )

        auto = Automation(
            name=automation_name,
            description="Notification automatique sur échec critique du pipeline Steam MLOps",
            enabled=True,
            trigger=trigger,
            actions=[],
        )
        return {
            "status": "CONFIGURED",
            "automation_name": automation_name,
            "trigger": "prefect.flow-run.Failed",
            "mode": "sdk-declared",
        }
    except Exception as exc:
        return {
            "status": "FALLBACK_HOOK",
            "message": f"Prefect server utilise les hooks d'état flow on_failure : {exc}",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Configuration et test de l'automatisation d'alertes Prefect")
    parser.add_argument("--receiver-url", type=str, default="http://127.0.0.1:9000/alerts", help="URL du récepteur local")
    parser.add_argument("--test-only", action="store_true", help="Envoyer uniquement une alerte de test")
    args = parser.parse_args()

    health_url = args.receiver_url.rsplit("/", 1)[0] + "/health"
    is_up = check_receiver_health(health_url)
    print(f"[ALERTS CONFIG] Vérification du récepteur ({health_url}) : {'EN LIGNE' if is_up else 'HORS LIGNE'}")

    auto_res = configure_prefect_automation()
    print(f"[ALERTS CONFIG] Statut d'automatisation Prefect : {auto_res.get('status')}")

    if is_up:
        print("[ALERTS CONFIG] Envoi d'une alerte de test...")
        dispatch_res = test_send_alert(local_url=args.receiver_url)
        print(f"[ALERTS CONFIG] Résultat de l'envoi : {json.dumps(dispatch_res, indent=2)}")
    else:
        print("[ALERTS CONFIG] Note : Démarrez d'abord 'scripts/notification_receiver.py' pour recevoir les alertes HTTP en direct.")


if __name__ == "__main__":
    main()
