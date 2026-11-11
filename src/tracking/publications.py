"""src/tracking/publications.py - Journal transactionnel durable des publications et promotions.

Fournit :
- Persistance relationnelle SQLite des opérations de publication du Model Registry.
- Enregistrement incrémental par jalons pour garantir la reprise après panne.
- Journalisation d'audit des bascules d'alias (promotions et retours arrière) dans `selection_events`.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional
from uuid import uuid4

DEFAULT_PUBLICATIONS_DB_PATH = Path(__file__).resolve().parent.parent.parent / "state" / "publications.db"

INIT_PUBLICATIONS_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS publications (
    publication_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    verdict TEXT NOT NULL,
    status TEXT NOT NULL,
    target_alias TEXT NOT NULL,
    regression_model_name TEXT,
    regression_model_uri TEXT,
    regression_version TEXT,
    regression_status TEXT DEFAULT 'PENDING',
    classification_model_name TEXT,
    classification_model_uri TEXT,
    classification_version TEXT,
    classification_status TEXT DEFAULT 'PENDING',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS selection_events (
    event_id TEXT PRIMARY KEY,
    publication_id TEXT,
    model_name TEXT NOT NULL,
    target_version TEXT NOT NULL,
    alias TEXT NOT NULL,
    previous_version TEXT,
    action TEXT NOT NULL, -- PROMOTION ou ROLLBACK
    reason TEXT NOT NULL,
    triggered_by TEXT NOT NULL DEFAULT 'coordinator',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pub_status ON publications(status);
CREATE INDEX IF NOT EXISTS idx_events_model ON selection_events(model_name);
"""


class PublicationManager:
    """Gestionnaire de persistance pour les transactions de publication et de promotion."""

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        self.db_path = Path(db_path) if db_path else DEFAULT_PUBLICATIONS_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=20.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.executescript(INIT_PUBLICATIONS_SQL)

    def create_publication(
        self,
        run_id: str,
        verdict: str,
        target_alias: str = "champion",
        publication_id: Optional[str] = None,
        regression_model_name: Optional[str] = None,
        classification_model_name: Optional[str] = None,
        regression_model_uri: Optional[str] = None,
        classification_model_uri: Optional[str] = None,
    ) -> str:
        """Déclare une nouvelle tentative de publication à l'état PENDING."""
        pub_id = publication_id or f"pub-{uuid4().hex[:8]}"
        now_iso = datetime.now(timezone.utc).isoformat()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO publications (
                    publication_id, run_id, verdict, status, target_alias,
                    regression_model_name, regression_model_uri, regression_status,
                    classification_model_name, classification_model_uri, classification_status,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING', ?, ?, 'PENDING', ?, ?)
                """,
                (
                    pub_id,
                    run_id,
                    verdict,
                    "PENDING",
                    target_alias,
                    regression_model_name,
                    regression_model_uri,
                    classification_model_name,
                    classification_model_uri,
                    now_iso,
                    now_iso,
                ),
            )
        return pub_id

    def update_publication(
        self,
        publication_id: str,
        *,
        status: Optional[str] = None,
        regression_version: Optional[str] = None,
        regression_status: Optional[str] = None,
        classification_version: Optional[str] = None,
        classification_status: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Met à jour un jalon spécifique de la publication."""
        now_iso = datetime.now(timezone.utc).isoformat()
        updates = ["updated_at = ?"]
        params: List[Any] = [now_iso]

        if status is not None:
            updates.append("status = ?")
            params.append(status)
        if regression_version is not None:
            updates.append("regression_version = ?")
            params.append(regression_version)
        if regression_status is not None:
            updates.append("regression_status = ?")
            params.append(regression_status)
        if classification_version is not None:
            updates.append("classification_version = ?")
            params.append(classification_version)
        if classification_status is not None:
            updates.append("classification_status = ?")
            params.append(classification_status)
        if error_message is not None:
            updates.append("error_message = ?")
            params.append(error_message)

        params.append(publication_id)
        sql = f"UPDATE publications SET {', '.join(updates)} WHERE publication_id = ?"

        with self._get_connection() as conn:
            conn.execute(sql, tuple(params))

    def record_selection_event(
        self,
        model_name: str,
        target_version: str,
        alias: str,
        previous_version: Optional[str] = None,
        action: str = "PROMOTION",
        reason: str = "Promotion après qualification",
        triggered_by: str = "coordinator",
        publication_id: Optional[str] = None,
    ) -> str:
        """Enregistre un événement auditable de promotion ou de rollback."""
        evt_id = f"evt-{uuid4().hex[:8]}"
        now_iso = datetime.now(timezone.utc).isoformat()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO selection_events (
                    event_id, publication_id, model_name, target_version, alias,
                    previous_version, action, reason, triggered_by, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evt_id,
                    publication_id,
                    model_name,
                    target_version,
                    alias,
                    previous_version,
                    action,
                    reason,
                    triggered_by,
                    now_iso,
                ),
            )
        return evt_id

    def get_publication(self, publication_id: str) -> Optional[Dict[str, Any]]:
        """Retourne les informations détaillées d'une publication."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM publications WHERE publication_id = ?", (publication_id,)
            ).fetchone()
            return dict(row) if row else None

    def list_publications(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Liste les publications enregistrées, avec filtre de statut optionnel."""
        with self._get_connection() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM publications WHERE status = ? ORDER BY created_at DESC",
                    (status,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM publications ORDER BY created_at DESC").fetchall()
            return [dict(r) for r in rows]

    def list_selection_events(self, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Liste les événements de sélection/promotion/rollback ordonnés chronologiquement."""
        with self._get_connection() as conn:
            if model_name:
                rows = conn.execute(
                    "SELECT * FROM selection_events WHERE model_name = ? ORDER BY created_at DESC",
                    (model_name,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM selection_events ORDER BY created_at DESC").fetchall()
            return [dict(r) for r in rows]
