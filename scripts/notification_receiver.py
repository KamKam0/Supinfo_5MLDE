"""scripts/notification_receiver.py - Récepteur local autonome d'alertes webhook.

Micro-serveur HTTP local (port 9000 par défaut) assurant la réception contrôlée,
la validation et l'archivage persistant des alertes émises par les pipelines Prefect.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import logging
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

DEFAULT_ALERTS_FILE = Path(__file__).resolve().parent.parent / "state" / "alerts_history.json"


class AlertStore:
    """Gestionnaire de persistance thread-safe des alertes reçues."""

    def __init__(self, storage_path: Optional[Path | str] = None) -> None:
        self.storage_path = Path(storage_path or DEFAULT_ALERTS_FILE).resolve()
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._alerts: List[Dict[str, Any]] = []
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        with self._lock:
            if self.storage_path.exists():
                try:
                    with open(self.storage_path, "r", encoding="utf-8") as f:
                        self._alerts = json.load(f)
                except Exception:
                    self._alerts = []
            else:
                self._alerts = []

    def _save_to_disk(self) -> None:
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self._alerts, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.error(f"[RECEIVER] Échec de sauvegarde sur disque : {exc}")

    def add_alert(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            alert_entry = {
                "receiver_id": f"rec-{uuid4().hex[:8]}",
                "received_at": datetime.now(timezone.utc).isoformat(),
                "payload": payload,
            }
            self._alerts.append(alert_entry)
            self._save_to_disk()
            return alert_entry

    def get_alerts(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._alerts)

    def clear(self) -> int:
        with self._lock:
            count = len(self._alerts)
            self._alerts = []
            self._save_to_disk()
            return count


GLOBAL_ALERT_STORE = AlertStore()


class NotificationHTTPRequestHandler(BaseHTTPRequestHandler):
    """Handler HTTP traitant les requêtes webhook d'alertes."""

    def _send_json(self, status_code: int, data: Dict[str, Any]) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/health", "/"):
            self._send_json(200, {"status": "UP", "service": "NotificationReceiver", "port": self.server.server_port})
        elif self.path in ("/alerts", "/alerts/"):
            alerts = GLOBAL_ALERT_STORE.get_alerts()
            self._send_json(200, {"count": len(alerts), "alerts": alerts})
        else:
            self._send_json(404, {"error": "Endpoint non trouvé"})

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length) if content_length > 0 else b""

        if self.path in ("/alerts", "/alerts/", "/webhook", "/webhook/"):
            try:
                payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
            except Exception as exc:
                self._send_json(400, {"error": f"JSON invalide : {exc}"})
                return

            entry = GLOBAL_ALERT_STORE.add_alert(payload)
            print(f"[RECEIVER] Alerte reçue et archivée : {entry['receiver_id']} (flow: {payload.get('flow_name', 'inconnu')})")
            self._send_json(200, {"status": "received", "entry": entry})

        elif self.path in ("/clear", "/clear/"):
            cleared = GLOBAL_ALERT_STORE.clear()
            self._send_json(200, {"status": "cleared", "count": cleared})
        else:
            self._send_json(404, {"error": "Endpoint non trouvé"})

    def do_DELETE(self) -> None:
        if self.path in ("/alerts", "/alerts/"):
            cleared = GLOBAL_ALERT_STORE.clear()
            self._send_json(200, {"status": "cleared", "count": cleared})
        else:
            self._send_json(404, {"error": "Endpoint non trouvé"})

    def log_message(self, format: str, *args: Any) -> None:
        # Masquer les requêtes de routine de la sortie standard
        pass


def run_receiver_server(host: str = "127.0.0.1", port: int = 9000, storage_path: Optional[Path | str] = None) -> HTTPServer:
    """Démarre le serveur HTTP de réception d'alertes."""
    if storage_path:
        global GLOBAL_ALERT_STORE
        GLOBAL_ALERT_STORE = AlertStore(storage_path=storage_path)

    server = HTTPServer((host, port), NotificationHTTPRequestHandler)
    print(f"[RECEIVER] Serveur d'alertes démarré sur http://{host}:{port}/")
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Récepteur d'alertes webhook local pour Prefect MLOps")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Hôte d'écoute")
    parser.add_argument("--port", type=int, default=9000, help="Port d'écoute")
    parser.add_argument("--storage", type=str, default=None, help="Chemin vers le fichier JSON de persistance")
    args = parser.parse_args()

    server = run_receiver_server(host=args.host, port=args.port, storage_path=args.storage)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[RECEIVER] Arrêt propre du serveur d'alertes.")
        server.server_close()


if __name__ == "__main__":
    main()
