"""tests/test_environment.py - Tests de validation de l'environnement et de l'infrastructure."""

import importlib.metadata
import os
import platform
import socket
import sys


def test_python_version():
    """Vérifie que la version majeure/mineure de Python est 3.12."""
    assert sys.version_info.major == 3
    assert sys.version_info.minor == 12


def test_core_package_versions():
    """Vérifie la conformité des versions des paquets critiques figés."""
    assert importlib.metadata.version("pandas") == "2.2.3"
    assert importlib.metadata.version("numpy") == "2.2.6"
    assert importlib.metadata.version("scikit-learn") == "1.6.1"
    assert importlib.metadata.version("joblib") == "1.5.1"
    assert importlib.metadata.version("great-expectations") == "1.23.2"
    assert importlib.metadata.version("mlflow") == "3.16.1"
    assert importlib.metadata.version("prefect") == "3.8.7"
    assert importlib.metadata.version("locust") == "2.46.6"


def test_infrastructure_ports():
    """Vérifie la disponibilité des ports de services d'infrastructure."""
    ports = {
        "postgres": int(os.getenv("POSTGRES_PORT", "5432")),
        "mlflow": int(os.getenv("MLFLOW_PORT", "7281")),
        "prefect": int(os.getenv("PREFECT_PORT", "4200")),
        "grafana": int(os.getenv("GRAFANA_PORT", "3001")),
    }
    for svc_name, port in ports.items():
        with socket.create_connection(("127.0.0.1", port), timeout=2.0) as sock:
            assert sock is not None, f"Service {svc_name} sur port {port} non joignable"
