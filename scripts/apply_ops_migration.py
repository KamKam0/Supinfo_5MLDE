"""scripts/apply_ops_migration.py - Application de la migration d'observabilité OPS.

Applique 'infra/postgres/migrations/020-ops-observability.sql' sur PostgreSQL 18.6
en tant que superuser (postgres) pour initialiser le schéma 'ops' et les rôles associés.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import psycopg

from src.settings import settings


def apply_migration() -> None:
    migration_path = settings.PROJECT_ROOT / "infra" / "postgres" / "migrations" / "020-ops-observability.sql"
    if not migration_path.exists():
        print(f"[ERROR] Fichier de migration introuvable : {migration_path}", file=sys.stderr)
        sys.exit(1)

    sql_content = migration_path.read_text(encoding="utf-8")

    # Connexion en superuser pour la création de rôles et privilèges
    conn_str = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@127.0.0.1:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"

    print(f"[MIGRATION] Connexion à {settings.POSTGRES_DB} sur port {settings.POSTGRES_PORT}...")
    try:
        with psycopg.connect(conn_str, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(sql_content)
        print(f"[MIGRATION] Succès : Schéma 'ops' et tables déployés avec succès.")
    except Exception as exc:
        print(f"[ERROR] Échec de l'application de la migration : {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    apply_migration()
