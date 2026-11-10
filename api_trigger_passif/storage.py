"""api_trigger_passif/storage.py - Persistance relationnelle SQLite et réservation atomique.

Fournit :
- Schéma relationnel SQLite avec contraintes d'unicité strictes.
- Transactions immédiates (BEGIN IMMEDIATE) pour garantir l'atomicité et l'isolation.
- Résolution idempotente des doublons (HTTP 200 sur même payload, 409 sur conflit).
- Constitution automatique d'un lot de 500 jeux dès que le seuil de non-affectés est atteint.
- Règle d'exclusivité : un appid n'appartient qu'à un seul lot (clé UNIQUE dans batch_items).
"""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import pandas as pd

from api_trigger_passif.validation import PushEventPayload


INIT_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    payload_sha256 TEXT NOT NULL,
    received_at TEXT NOT NULL,
    source TEXT NOT NULL,
    raw_payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS games (
    appid INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    release_date TEXT,
    required_age INTEGER NOT NULL DEFAULT 0,
    price REAL NOT NULL DEFAULT 0.0,
    dlc_count INTEGER NOT NULL DEFAULT 0,
    windows INTEGER NOT NULL DEFAULT 1,
    mac INTEGER NOT NULL DEFAULT 0,
    linux INTEGER NOT NULL DEFAULT 0,
    achievements INTEGER NOT NULL DEFAULT 0,
    genres TEXT NOT NULL DEFAULT '[]',
    positive INTEGER NOT NULL DEFAULT 0,
    negative INTEGER NOT NULL DEFAULT 0,
    pct_pos_total REAL NOT NULL,
    num_reviews_total INTEGER NOT NULL,
    estimated_owners TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS batches (
    batch_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    failure_reason TEXT,
    snapshot_path TEXT,
    snapshot_sha256 TEXT,
    row_count INTEGER NOT NULL DEFAULT 500
);

CREATE TABLE IF NOT EXISTS batch_items (
    batch_id TEXT NOT NULL REFERENCES batches(batch_id) ON DELETE CASCADE,
    appid INTEGER NOT NULL UNIQUE REFERENCES games(appid) ON DELETE RESTRICT,
    assigned_at TEXT NOT NULL,
    PRIMARY KEY (batch_id, appid)
);

CREATE INDEX IF NOT EXISTS idx_games_created_at ON games(created_at);
CREATE INDEX IF NOT EXISTS idx_batches_status ON batches(status);
"""


class PushStorage:
    """Gestionnaire de persistance SQLite pour la collecte passive."""

    THRESHOLD_BATCH_SIZE: int = 500

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        if db_path:
            self.db_path = Path(db_path).resolve()
        else:
            self.db_path = (Path(__file__).parent.parent / "state" / "push_collector.db").resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.executescript(INIT_SQL)

    def record_event(
        self,
        event: PushEventPayload,
        raw_bytes: bytes,
    ) -> Tuple[int, Dict[str, Any]]:
        """Enregistre un événement avec garantie d'atomicité et détection de seuil.

        Returns:
            Tuple (http_code, response_dict)
            - 202 : accepté (avec création éventuelle de lot)
            - 200 : idempotent (déjà reçu à l'identique)
            - 409 : conflit (même event_id mais contenu différent)
        """
        payload_sha = hashlib.sha256(raw_bytes).hexdigest()
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        game = event.game

        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE;")

            # 1. Vérification de l'idempotence sur event_id
            cursor = conn.execute("SELECT payload_sha256 FROM events WHERE event_id = ?", (event.event_id,))
            existing_event = cursor.fetchone()

            if existing_event:
                if existing_event["payload_sha256"] == payload_sha:
                    # Même événement à l'identique -> 200 OK (idempotence pure)
                    conn.execute("COMMIT;")
                    return 200, {
                        "status": "idempotent_duplicate",
                        "event_id": event.event_id,
                        "message": "Événement identique déjà reçu et persistant.",
                    }
                else:
                    # Même identifiant mais contenu différent -> 409 Conflict
                    conn.execute("ROLLBACK;")
                    return 409, {
                        "error": "Conflict",
                        "message": f"L'event_id '{event.event_id}' existe déjà avec un payload différent.",
                    }

            # 2. Insertion dans la table events
            conn.execute(
                """
                INSERT INTO events (event_id, payload_sha256, received_at, source, raw_payload)
                VALUES (?, ?, ?, ?, ?)
                """,
                (event.event_id, payload_sha, now_iso, event.source, raw_bytes.decode("utf-8")),
            )

            # 3. Insertion ou mise à jour du jeu dans la table games
            conn.execute(
                """
                INSERT INTO games (
                    appid, name, release_date, required_age, price, dlc_count,
                    windows, mac, linux, achievements, genres, positive, negative,
                    pct_pos_total, num_reviews_total, estimated_owners, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(appid) DO UPDATE SET
                    name = excluded.name,
                    release_date = excluded.release_date,
                    required_age = excluded.required_age,
                    price = excluded.price,
                    dlc_count = excluded.dlc_count,
                    windows = excluded.windows,
                    mac = excluded.mac,
                    linux = excluded.linux,
                    achievements = excluded.achievements,
                    genres = excluded.genres,
                    positive = excluded.positive,
                    negative = excluded.negative,
                    pct_pos_total = excluded.pct_pos_total,
                    num_reviews_total = excluded.num_reviews_total,
                    estimated_owners = excluded.estimated_owners
                WHERE appid NOT IN (SELECT appid FROM batch_items);
                """,
                (
                    game.appid,
                    game.name,
                    game.release_date,
                    game.required_age,
                    game.price,
                    game.dlc_count,
                    1 if game.windows else 0,
                    1 if game.mac else 0,
                    1 if game.linux else 0,
                    game.achievements,
                    game.genres if isinstance(game.genres, str) else str(game.genres),
                    game.positive,
                    game.negative,
                    game.pct_pos_total,
                    game.num_reviews_total,
                    game.estimated_owners,
                    now_iso,
                ),
            )

            # 4. Vérification du seuil de 500 jeux non affectés
            cur_count = conn.execute(
                "SELECT count(*) as cnt FROM games WHERE appid NOT IN (SELECT appid FROM batch_items)"
            )
            unassigned_count = cur_count.fetchone()["cnt"]

            created_batch_id: Optional[str] = None
            if unassigned_count >= self.THRESHOLD_BATCH_SIZE:
                # Constitution atomique d'un lot de 500 jeux
                cur_unassigned = conn.execute(
                    """
                    SELECT appid FROM games 
                    WHERE appid NOT IN (SELECT appid FROM batch_items)
                    ORDER BY created_at ASC, appid ASC
                    LIMIT ?
                    """,
                    (self.THRESHOLD_BATCH_SIZE,),
                )
                appids_to_batch = [row["appid"] for row in cur_unassigned.fetchall()]

                created_batch_id = f"batch-push-{uuid4().hex[:8]}"
                conn.execute(
                    """
                    INSERT INTO batches (batch_id, status, created_at, row_count)
                    VALUES (?, 'PENDING', ?, ?)
                    """,
                    (created_batch_id, now_iso, len(appids_to_batch)),
                )

                for appid in appids_to_batch:
                    conn.execute(
                        """
                        INSERT INTO batch_items (batch_id, appid, assigned_at)
                        VALUES (?, ?, ?)
                        """,
                        (created_batch_id, appid, now_iso),
                    )

                remaining_count = unassigned_count - len(appids_to_batch)
                conn.execute("COMMIT;")
                return 202, {
                    "status": "accepted",
                    "event_id": event.event_id,
                    "batch_created": True,
                    "batch_id": created_batch_id,
                    "batch_size": len(appids_to_batch),
                    "unassigned_remaining": remaining_count,
                }

            conn.execute("COMMIT;")
            return 202, {
                "status": "accepted",
                "event_id": event.event_id,
                "batch_created": False,
                "batch_id": None,
                "unassigned_count": unassigned_count,
            }

        except Exception as exc:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise exc
        finally:
            conn.close()

    def get_status(self) -> Dict[str, Any]:
        """Retourne la situation globale de la collecte passive."""
        with self._get_connection() as conn:
            ev_cnt = conn.execute("SELECT count(*) as cnt FROM events").fetchone()["cnt"]
            unassigned_cnt = conn.execute(
                "SELECT count(*) as cnt FROM games WHERE appid NOT IN (SELECT appid FROM batch_items)"
            ).fetchone()["cnt"]
            batches_cnt = conn.execute("SELECT count(*) as cnt FROM batches").fetchone()["cnt"]
            
            recent_batches = [
                dict(row)
                for row in conn.execute(
                    "SELECT batch_id, status, created_at, completed_at, row_count, failure_reason FROM batches ORDER BY created_at DESC LIMIT 10"
                ).fetchall()
            ]

        return {
            "events_count": ev_cnt,
            "unassigned_games_count": unassigned_cnt,
            "threshold": self.THRESHOLD_BATCH_SIZE,
            "batches_count": batches_cnt,
            "recent_batches": recent_batches,
        }

    def get_batch(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Retourne le détail d'un lot et la liste des identifiants associés."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM batches WHERE batch_id = ?", (batch_id,))
            batch_row = cur.fetchone()
            if not batch_row:
                return None

            items = conn.execute(
                "SELECT appid FROM batch_items WHERE batch_id = ? ORDER BY appid ASC", (batch_id,)
            ).fetchall()

        res = dict(batch_row)
        res["appids"] = [row["appid"] for row in items]
        return res

    def get_batch_games_df(self, batch_id: str) -> Optional[pd.DataFrame]:
        """Extrait un DataFrame des 500 jeux composant le lot."""
        with self._get_connection() as conn:
            query = """
                SELECT g.* FROM games g
                JOIN batch_items bi ON g.appid = bi.appid
                WHERE bi.batch_id = ?
                ORDER BY g.appid ASC
            """
            df = pd.read_sql_query(query, conn, params=(batch_id,))
        if len(df) == 0:
            return None
        # Convertir les colonnes booléennes
        df["windows"] = df["windows"].astype(bool)
        df["mac"] = df["mac"].astype(bool)
        df["linux"] = df["linux"].astype(bool)
        return df

    def update_batch_status(
        self,
        batch_id: str,
        status: str,
        *,
        snapshot_path: Optional[str] = None,
        snapshot_sha256: Optional[str] = None,
        failure_reason: Optional[str] = None,
    ) -> None:
        """Met à jour le statut et les métadonnées d'exécution d'un lot."""
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self._get_connection() as conn:
            if status == "RUNNING":
                conn.execute(
                    "UPDATE batches SET status = ?, started_at = ? WHERE batch_id = ?",
                    (status, now_iso, batch_id),
                )
            elif status in ("SUCCEEDED", "FAILED"):
                conn.execute(
                    """
                    UPDATE batches SET status = ?, completed_at = ?, snapshot_path = ?,
                                       snapshot_sha256 = ?, failure_reason = ?
                    WHERE batch_id = ?
                    """,
                    (status, now_iso, snapshot_path, snapshot_sha256, failure_reason, batch_id),
                )
            else:
                conn.execute("UPDATE batches SET status = ? WHERE batch_id = ?", (status, batch_id))
