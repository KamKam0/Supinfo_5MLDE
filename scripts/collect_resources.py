"""scripts/collect_resources.py - Échantillonneur de ressources CPU/RAM.

Collecte chaque seconde l'utilisation des ressources système :
- CPU hôte (%) et RAM hôte (Mo, %)
- Processus Uvicorn (CPU %, RAM RSS Mo, nombre de workers)
- Processus Locust (CPU %, RAM RSS Mo)
- Conteneurs Docker sous-jacents (PostgreSQL)

Peut s'utiliser en module (classe ResourceMonitor) ou en script CLI autonome.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import psutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("resource_collector")


def find_target_processes(pattern: str) -> List[psutil.Process]:
    """Retrouve la liste des processus correspondant à un motif de ligne de commande."""
    matches = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = " ".join(proc.info.get("cmdline") or [])
            if pattern.lower() in cmdline.lower() and proc.pid != os.getpid():
                matches.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return matches


class ResourceMonitor:
    """Collecteur asynchrone sur thread d'arrière-plan avec enregistrement CSV horodaté."""

    def __init__(
        self,
        output_csv_path: Path,
        interval_seconds: float = 1.0,
        uvicorn_pids: Optional[List[int]] = None,
    ):
        self.output_csv_path = Path(output_csv_path)
        self.interval_seconds = interval_seconds
        self.target_uvicorn_pids = uvicorn_pids or []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.samples: List[Dict[str, Any]] = []

    def start(self) -> None:
        """Démarre le thread d'échantillonnage."""
        self.output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._stop_event.clear()
        self.samples = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, float]:
        """Arrête le thread, finalise l'écriture CSV et retourne le résumé statistique."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)

        return self.compute_summary()

    def _run(self) -> None:
        """Boucle d'échantillonnage périodique."""
        fieldnames = [
            "timestamp_utc",
            "system_cpu_pct",
            "system_ram_pct",
            "system_ram_used_mb",
            "uvicorn_cpu_pct",
            "uvicorn_rss_mb",
            "uvicorn_workers_count",
            "locust_cpu_pct",
            "locust_rss_mb",
        ]

        with open(self.output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            while not self._stop_event.is_set():
                now_iso = datetime.now(timezone.utc).isoformat()

                # 1. Métriques Hôte
                sys_cpu = psutil.cpu_percent(interval=None)
                sys_ram = psutil.virtual_memory()

                # 2. Processus Uvicorn
                uvicorn_cpu = 0.0
                uvicorn_rss = 0.0
                uvicorn_count = 0

                uvicorn_procs = []
                if self.target_uvicorn_pids:
                    for pid in self.target_uvicorn_pids:
                        try:
                            uvicorn_procs.append(psutil.Process(pid))
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                else:
                    uvicorn_procs = find_target_processes("api_prediction.main")

                for p in uvicorn_procs:
                    try:
                        uvicorn_cpu += p.cpu_percent(interval=None)
                        uvicorn_rss += p.memory_info().rss / (1024 * 1024)
                        uvicorn_count += 1
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                # 3. Processus Locust
                locust_cpu = 0.0
                locust_rss = 0.0
                locust_procs = find_target_processes("locust")
                for p in locust_procs:
                    try:
                        locust_cpu += p.cpu_percent(interval=None)
                        locust_rss += p.memory_info().rss / (1024 * 1024)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                sample = {
                    "timestamp_utc": now_iso,
                    "system_cpu_pct": round(sys_cpu, 2),
                    "system_ram_pct": round(sys_ram.percent, 2),
                    "system_ram_used_mb": round(sys_ram.used / (1024 * 1024), 2),
                    "uvicorn_cpu_pct": round(uvicorn_cpu, 2),
                    "uvicorn_rss_mb": round(uvicorn_rss, 2),
                    "uvicorn_workers_count": uvicorn_count,
                    "locust_cpu_pct": round(locust_cpu, 2),
                    "locust_rss_mb": round(locust_rss, 2),
                }

                self.samples.append(sample)
                writer.writerow(sample)
                csvfile.flush()

                self._stop_event.wait(self.interval_seconds)

    def compute_summary(self) -> Dict[str, float]:
        """Calcule la synthèse statistique des échantillons collectés."""
        if not self.samples:
            return {
                "avg_system_cpu_pct": 0.0,
                "max_system_cpu_pct": 0.0,
                "avg_uvicorn_cpu_pct": 0.0,
                "max_uvicorn_cpu_pct": 0.0,
                "avg_uvicorn_rss_mb": 0.0,
                "max_uvicorn_rss_mb": 0.0,
                "avg_locust_cpu_pct": 0.0,
                "max_locust_cpu_pct": 0.0,
                "samples_count": 0,
            }

        n = len(self.samples)
        return {
            "avg_system_cpu_pct": round(sum(s["system_cpu_pct"] for s in self.samples) / n, 2),
            "max_system_cpu_pct": round(max(s["system_cpu_pct"] for s in self.samples), 2),
            "avg_uvicorn_cpu_pct": round(sum(s["uvicorn_cpu_pct"] for s in self.samples) / n, 2),
            "max_uvicorn_cpu_pct": round(max(s["uvicorn_cpu_pct"] for s in self.samples), 2),
            "avg_uvicorn_rss_mb": round(sum(s["uvicorn_rss_mb"] for s in self.samples) / n, 2),
            "max_uvicorn_rss_mb": round(max(s["uvicorn_rss_mb"] for s in self.samples), 2),
            "avg_locust_cpu_pct": round(sum(s["locust_cpu_pct"] for s in self.samples) / n, 2),
            "max_locust_cpu_pct": round(max(s["locust_cpu_pct"] for s in self.samples), 2),
            "samples_count": n,
        }


def main():
    parser = argparse.ArgumentParser(description="Collecteur de métriques d'infrastructure en continu")
    parser.add_argument("--output", type=str, default="reports/load/metrics.csv", help="Chemin du fichier CSV de sortie")
    parser.add_argument("--duration", type=int, default=30, help="Durée de collecte en secondes")
    parser.add_argument("--interval", type=float, default=1.0, help="Intervalle d'échantillonnage en secondes")
    args = parser.parse_args()

    monitor = ResourceMonitor(output_csv_path=Path(args.output), interval_seconds=args.interval)
    print(f"[RESOURCES] Démarrage de la collecte vers {args.output} pour {args.duration}s...")
    monitor.start()
    try:
        time.sleep(args.duration)
    finally:
        summary = monitor.stop()
        print("\n[RESOURCES] Résumé des métriques collectées :")
        for k, v in summary.items():
            print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
