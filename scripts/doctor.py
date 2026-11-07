#!/usr/bin/env python3
"""scripts/doctor.py - Outil autonome de diagnostic d'environnement et de conformité MLOps.

Vérifie l'environnement Python, les versions des bibliothèques, la configuration des ports
et la connectivité aux services d'infrastructure (PostgreSQL, MLflow, Prefect, Grafana).
Aucun secret ni mot de passe n'est affiché en clair.
Génère le rapport canonique `reports/environment.json`.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path


def mask_secret(value: str | None) -> str:
    """Masque une valeur sensible pour éviter toute fuite de secret."""
    if not value:
        return "[NON DÉFINI]"
    if len(value) <= 4:
        return "***"
    return f"{value[:2]}***{value[-2:]}"


def check_port(host: str, port: int, timeout: float = 1.5) -> bool:
    """Vérifie si un port réseau TCP est ouvert et répond."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefException if "ConnectionRefException" in globals() else OSError):
        return False


def get_pkg_version(name: str) -> str:
    """Récupère la version installée d'un paquet Python."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NON_INSTALLE"


def check_postgres_connection(
    host: str, port: int, dbname: str, user: str, password: str | None
) -> dict[str, str | bool]:
    """Tente une connexion SQL réelle à PostgreSQL sans journaliser le mot de passe."""
    try:
        import psycopg

        conn_str = f"host={host} port={port} dbname={dbname} user={user}"
        if password:
            conn_str += f" password={password}"

        with psycopg.connect(conn_str, connect_timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                res = cur.fetchone()
                # Vérifier si le schéma tracing existe
                cur.execute(
                    "SELECT EXISTS(SELECT 1 FROM information_schema.schemata WHERE schema_name = 'tracing');"
                )
                tracing_exists = bool(cur.fetchone()[0])
                return {
                    "connected": True,
                    "ping": res == (1,),
                    "tracing_schema_initialized": tracing_exists,
                    "message": "Connexion PostgreSQL opérationnelle",
                }
    except Exception as exc:
        return {
            "connected": False,
            "ping": False,
            "tracing_schema_initialized": False,
            "message": f"Non joignable ({exc.__class__.__name__})",
        }


def run_diagnostics() -> dict:
    """Exécute l'ensemble des contrôles d'intégrité de la plateforme."""
    report_time = datetime.now(timezone.utc).isoformat()
    project_root = Path(__file__).resolve().parent.parent

    # Lecture des variables de configuration
    env_file = project_root / ".env"
    env_vars: dict[str, str] = {}
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env_vars[k.strip()] = v.strip()

    postgres_port = int(env_vars.get("POSTGRES_PORT") or os.getenv("POSTGRES_PORT") or 5432)
    mlflow_port = int(env_vars.get("MLFLOW_PORT") or os.getenv("MLFLOW_PORT") or 7281)
    prefect_port = int(env_vars.get("PREFECT_PORT") or os.getenv("PREFECT_PORT") or 4200)
    grafana_port = int(env_vars.get("GRAFANA_PORT") or os.getenv("GRAFANA_PORT") or 3000)

    pg_user = env_vars.get("POSTGRES_USER") or os.getenv("POSTGRES_USER") or "postgres"
    pg_password = env_vars.get("POSTGRES_PASSWORD") or os.getenv("POSTGRES_PASSWORD") or "5mlde-local-postgres"
    pg_db = env_vars.get("POSTGRES_DB") or os.getenv("POSTGRES_DB") or "steam_api"

    # Vérification Python & Environnement
    python_info = {
        "executable": sys.executable,
        "version": platform.python_version(),
        "major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        "in_venv": sys.prefix != sys.base_prefix,
        "platform": platform.platform(),
        "architecture": platform.machine(),
    }

    # Vérification des paquets obligatoires
    core_packages = [
        "pandas",
        "numpy",
        "scikit-learn",
        "joblib",
        "great-expectations",
        "mlflow",
        "prefect",
        "fastapi",
        "uvicorn",
        "pydantic",
        "psycopg",
        "httpx",
        "locust",
        "pytest",
    ]
    package_versions = {pkg: get_pkg_version(pkg) for pkg in core_packages}

    # Vérification des ports réseau locaux
    ports_status = {
        "postgres": {"port": postgres_port, "reachable": check_port("127.0.0.1", postgres_port)},
        "mlflow": {"port": mlflow_port, "reachable": check_port("127.0.0.1", mlflow_port)},
        "prefect": {"port": prefect_port, "reachable": check_port("127.0.0.1", prefect_port)},
        "grafana": {"port": grafana_port, "reachable": check_port("127.0.0.1", grafana_port)},
    }

    # Connectivité PostgreSQL
    postgres_diag = check_postgres_connection(
        host="127.0.0.1",
        port=postgres_port,
        dbname=pg_db,
        user=pg_user,
        password=pg_password,
    )

    # Synthèse générale
    all_packages_installed = all(v != "NON_INSTALLE" for v in package_versions.values())
    python_312_valid = python_info["major_minor"] == "3.12"
    services_ready = all(s["reachable"] for s in ports_status.values()) and postgres_diag["connected"]

    overall_status = "READY" if (python_312_valid and all_packages_installed and services_ready) else "PARTIAL"

    report = {
        "timestamp": report_time,
        "status": overall_status,
        "system": python_info,
        "packages": package_versions,
        "ports": ports_status,
        "database": postgres_diag,
        "safe_config": {
            "postgres_host": "127.0.0.1",
            "postgres_port": postgres_port,
            "postgres_db": pg_db,
            "postgres_user": pg_user,
            "postgres_password": mask_secret(pg_password),
            "mlflow_tracking_uri": f"http://127.0.0.1:{mlflow_port}",
            "prefect_api_url": f"http://127.0.0.1:{prefect_port}/api",
            "grafana_url": f"http://127.0.0.1:{grafana_port}",
        },
    }

    # Sauvegarde dans reports/environment.json
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "environment.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def main() -> int:
    report = run_diagnostics()

    print("=" * 70)
    print("    DIAGNOSTIC D'ENVIRONNEMENT ET D'INFRASTRUCTURE (scripts/doctor.py)")
    print("=" * 70)
    print(f"Horodatage       : {report['timestamp']}")
    print(f"Statut Global    : {report['status']}")
    print(f"OS / Plateforme  : {report['system']['platform']} ({report['system']['architecture']})")
    print(f"Interpréteur     : Python {report['system']['version']} [vEnv: {report['system']['in_venv']}]")
    print("-" * 70)
    print("PAQUETS CORE MLOPS :")
    for pkg, ver in report["packages"].items():
        status_flag = "[OK]" if ver != "NON_INSTALLE" else "[ABSENT]"
        print(f"  {status_flag:<8} {pkg:<22} : {ver}")
    print("-" * 70)
    print("SERVICES & PORTS RÉSEAU (127.0.0.1) :")
    for svc, details in report["ports"].items():
        status_flag = "[ACTIF]" if details["reachable"] else "[INACTIF]"
        print(f"  {status_flag:<9} {svc.capitalize():<12} (Port {details['port']})")
    print("-" * 70)
    print("BASE POSTGRESQL (steam_api) :")
    pg = report["database"]
    print(f"  Connexion      : {'[SUCCÈS]' if pg['connected'] else '[ÉCHEC]'}")
    print(f"  Message        : {pg['message']}")
    if pg["connected"]:
        print(f"  Schéma tracing : {'[PRÉSENT]' if pg['tracing_schema_initialized'] else '[NON INITIALISÉ]'}")
    print("-" * 70)
    print("Rapport exporté : reports/environment.json")
    print("=" * 70)

    # Le script retourne 0 si l'environnement Python et paquets sont OK
    python_and_pkgs_ok = (
        report["system"]["major_minor"] == "3.12"
        and all(v != "NON_INSTALLE" for v in report["packages"].values())
    )
    return 0 if python_and_pkgs_ok else 1


if __name__ == "__main__":
    sys.exit(main())
