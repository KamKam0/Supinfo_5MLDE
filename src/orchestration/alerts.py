"""src/orchestration/alerts.py - Système d'alertes locales obligatoires et connecteur Teams conditionnel.

Fournit :
- Le schéma Pydantic typé des messages d'alerte (AlertMessage).
- Le formateur de charge utile Microsoft Teams (Adaptive Card / MessageCard).
- Le dispatcheur résilient d'alertes (AlertDispatcher) :
  1. Envoi systématique au récepteur local (port 9000).
  2. Envoi externe vers Microsoft Teams STRICTEMENT conditionné à l'autorisation explicite de l'utilisateur.
  3. Gestion d'erreur sans propagation d'exception bloquante.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
from typing import Any, Dict, Optional
import httpx
from pydantic import BaseModel, Field

from src.settings import settings

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_WEBHOOK_URL = "http://127.0.0.1:9000/alerts"


class AlertMessage(BaseModel):
    """Contrat de données structuré pour les alertes de pipeline MLOps."""

    flow_name: str
    flow_run_id: str
    state: str = "FAILED"
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    error_message: str
    environment: str = "local"
    mode: str = "training"
    batch_id: Optional[str] = None
    mlflow_run_id: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


def format_teams_card_payload(alert: AlertMessage) -> Dict[str, Any]:
    """Formate une carte adaptative Microsoft Teams conforme au standard Webhook Workflows."""
    return {
        "@type": "MessageCard",
        "@context": "https://schema.org/extensions",
        "summary": f"Alerte MLOps : Échec du flow {alert.flow_name}",
        "themeColor": "D63031",  # Rouge alerte
        "title": f"🚨 [MLOPS ALERT] Échec critique du pipeline '{alert.flow_name}'",
        "sections": [
            {
                "activityTitle": f"Flow Run : `{alert.flow_run_id}`",
                "activitySubtitle": f"Horodatage UTC : {alert.timestamp}",
                "facts": [
                    {"name": "Environnement :", "value": alert.environment},
                    {"name": "Mode d'exécution :", "value": alert.mode},
                    {"name": "Lot de données (batch_id) :", "value": alert.batch_id or "Non disponible"},
                    {"name": "Run MLflow associé :", "value": alert.mlflow_run_id or "Non disponible"},
                    {"name": "État terminal :", "value": alert.state},
                    {"name": "Cause de l'erreur :", "value": alert.error_message},
                ],
                "markdown": True,
            }
        ],
    }


class AlertDispatcher:
    """Dispatcheur d'alertes assurant la notification locale et conditionnelle externe."""

    def __init__(
        self,
        local_webhook_url: Optional[str] = None,
        teams_webhook_url: Optional[str] = None,
        teams_enabled: Optional[bool] = None,
        timeout: float = 5.0,
    ) -> None:
        self.local_webhook_url = local_webhook_url or os.environ.get("LOCAL_ALERT_WEBHOOK_URL", DEFAULT_LOCAL_WEBHOOK_URL)
        self.teams_webhook_url = teams_webhook_url or os.environ.get("TEAMS_WEBHOOK_URL")
        if teams_enabled is not None:
            self.teams_enabled = teams_enabled
        else:
            self.teams_enabled = os.environ.get("TEAMS_ALERT_ENABLED", "false").lower() in ("true", "1", "yes")
        self.timeout = timeout

    def send_alert(self, alert: AlertMessage) -> Dict[str, Any]:
        """Transmet l'alerte au récepteur local obligatoire et à Teams si autorisé.

        Règle d'or : Tout échec d'envoi n'interrompt ni ne corrompt le job métier amont.
        """
        results: Dict[str, Any] = {
            "flow_run_id": alert.flow_run_id,
            "timestamp": alert.timestamp,
            "local": {"sent": False, "status": "UNKNOWN"},
            "teams": {"sent": False, "status": "SKIPPED"},
        }

        # 1. Envoi au récepteur HTTP local autonome
        if self.local_webhook_url:
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(self.local_webhook_url, json=alert.model_dump())
                    if resp.status_code in (200, 201, 202):
                        results["local"] = {"sent": True, "status": "DELIVERED", "code": resp.status_code}
                        logger.info(f"[ALERT] Alerte transmise avec succès au récepteur local : {self.local_webhook_url}")
                    else:
                        results["local"] = {"sent": False, "status": "REJECTED", "code": resp.status_code}
                        logger.warning(f"[ALERT] Récepteur local a retourné HTTP {resp.status_code}")
            except Exception as exc:
                results["local"] = {"sent": False, "status": "CONNECTION_FAILED", "error": str(exc)}
                logger.warning(f"[ALERT] Impossible de joindre le récepteur local ({self.local_webhook_url}) : {exc}")

        # 2. Envoi conditionnel Teams
        # RÈGLE ABSOLUE : Zéro envoi externe sans canal configuré et autorisation formelle de l'opérateur
        if self.teams_enabled and self.teams_webhook_url:
            teams_payload = format_teams_card_payload(alert)
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(self.teams_webhook_url, json=teams_payload)
                    if resp.status_code in (200, 202):
                        results["teams"] = {"sent": True, "status": "DELIVERED", "code": resp.status_code}
                        logger.info("[ALERT] Alerte transmise avec succès au webhook Teams.")
                    else:
                        results["teams"] = {"sent": False, "status": "REJECTED", "code": resp.status_code}
                        logger.warning(f"[ALERT] Webhook Teams a répondu avec code {resp.status_code}")
            except Exception as exc:
                results["teams"] = {"sent": False, "status": "CONNECTION_FAILED", "error": str(exc)}
                logger.warning(f"[ALERT] Échec d'envoi webhook Teams : {exc}")
        else:
            reason = "Teams désactivé par défaut (TEAMS_ALERT_ENABLED=false)" if not self.teams_enabled else "URL webhook Teams non configurée"
            results["teams"] = {"sent": False, "status": "SKIPPED", "reason": reason}
            logger.info(f"[ALERT] Envoi Teams ignoré : {reason}")

        return results


def trigger_failure_alert(
    flow_name: str,
    flow_run_id: str,
    error_message: str,
    batch_id: Optional[str] = None,
    mlflow_run_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Helper global déclenchant immédiatement la notification d'échec."""
    alert = AlertMessage(
        flow_name=flow_name,
        flow_run_id=flow_run_id,
        state="FAILED",
        error_message=error_message,
        batch_id=batch_id,
        mlflow_run_id=mlflow_run_id,
        details=details or {},
    )
    dispatcher = AlertDispatcher()
    return dispatcher.send_alert(alert)
