"""scripts/run_capacity_matrix.py - Pilote automatisé de la matrice de tests de charge Uvicorn.

Exécute la matrice comparative de performance multi-workers (1, 2, 4 workers Uvicorn)
sous Locust 2.46.6 conformément au protocole de charge défini :
- Démarrage et arrêt contrôlés des processus Uvicorn (sans --reload).
- Échantillonnage continu des ressources CPU/RAM via ResourceMonitor.
- Exécution isolée des profils (Single, Batch 20, Mixte) sous Locust headless.
- Archivage complet et immuable dans reports/load/<campaign_id>/ (CSV Locust, métriques ressources, campaign.json).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.collect_resources import ResourceMonitor

from src.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("capacity_matrix")


def wait_for_service_ready(url: str, timeout_seconds: int = 30) -> bool:
    """Sonde l'endpoint /health/ready jusqu'à confirmation HTTP 200."""
    ready_url = f"{url.rstrip('/')}/health/ready"
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        try:
            with urllib.request.urlopen(ready_url, timeout=2) as res:
                if res.status == 200:
                    data = json.loads(res.read().decode("utf-8"))
                    if data.get("models_loaded") is True:
                        return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def start_uvicorn_server(workers: int, port: int = 8001) -> subprocess.Popen:
    """Démarre le serveur Uvicorn avec le nombre spécifié de workers sans rechargement."""
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "api_prediction.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--workers",
        str(workers),
        "--no-access-log",
    ]
    logger.info(f"[SERVER] Démarrage de Uvicorn ({workers} workers) sur port {port}...")
    proc = subprocess.Popen(
        cmd,
        cwd=str(settings.PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


def stop_uvicorn_server(proc: subprocess.Popen) -> None:
    """Arrête proprement le processus Uvicorn et ses sous-processus workers."""
    logger.info(f"[SERVER] Arrêt du processus Uvicorn (PID {proc.pid})...")
    try:
        proc.terminate()
        proc.wait(timeout=5.0)
    except Exception:
        proc.kill()


def run_locust_benchmark(
    campaign_id: str,
    scenario: str,
    workers_count: int,
    users: int,
    duration_seconds: int,
    output_dir: Path,
    host: str = "http://127.0.0.1:8001",
) -> Dict[str, Any]:
    """Exécute un tir de charge Locust unitaire et consigne les résultats."""
    campaign_dir = output_dir / campaign_id
    campaign_dir.mkdir(parents=True, exist_ok=True)

    user_class_map = {
        "single": "SinglePredictionUser",
        "batch": "BatchPredictionUser",
        "mixed": "MixedWorkloadUser",
    }
    user_class = user_class_map.get(scenario, "MixedWorkloadUser")

    # Démarrage de la collecte de ressources
    resource_csv = campaign_dir / "resources.csv"
    monitor = ResourceMonitor(output_csv_path=resource_csv, interval_seconds=1.0)
    monitor.start()

    logger.info(
        f"[LOAD] Lancement tir Locust ({scenario}, {users} users, {duration_seconds}s, "
        f"{workers_count} workers) -> {campaign_id}..."
    )

    env = os.environ.copy()
    env["LOCUST_CAMPAIGN_ID"] = campaign_id

    locust_csv_prefix = str(campaign_dir / "locust")
    locust_cmd = [
        sys.executable,
        "-m",
        "locust",
        "-f",
        "tests/load/locustfile.py",
        "--host",
        host,
        "--headless",
        "-u",
        str(users),
        "-r",
        str(max(2, users // 2)),
        "--run-time",
        f"{duration_seconds}s",
        "--csv",
        locust_csv_prefix,
        "--only-summary",
        user_class,
    ]

    t0 = time.time()
    try:
        res = subprocess.run(
            locust_cmd,
            cwd=str(settings.PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=duration_seconds + 30,
        )
        stdout = res.stdout
    except Exception as exc:
        stdout = f"ERROR: {exc}"
    actual_duration = round(time.time() - t0, 2)

    # Arrêt du moniteur de ressources
    resource_summary = monitor.stop()

    # Sauvegarde des métadonnées du tir (campaign.json)
    metadata = {
        "campaign_id": campaign_id,
        "scenario": scenario,
        "user_class": user_class,
        "workers_count": workers_count,
        "target_users": users,
        "planned_duration_seconds": duration_seconds,
        "actual_duration_seconds": actual_duration,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": host,
        "resources_summary": resource_summary,
        "stdout_tail": stdout[-500:] if stdout else "",
    }

    with open(campaign_dir / "campaign.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"[LOAD DONE] Tir {campaign_id} finalisé. Métadonnées archivées.")
    return metadata


def execute_capacity_matrix(
    workers_list: List[int],
    scenarios_list: List[str],
    concurrency_list: List[int],
    duration_seconds: int = 30,
    output_base_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Pilote la matrice complète de benchmarks et redémarre Uvicorn à chaque palier."""
    base_dir = output_base_dir or (settings.PROJECT_ROOT / "reports" / "load")
    base_dir.mkdir(parents=True, exist_ok=True)

    all_tirs = []

    for w in workers_list:
        logger.info(f"\n{'='*70}\n[PALIER WORKERS] Configuration : {w} WORKER(S) UVICORN\n{'='*70}")
        server_proc = start_uvicorn_server(workers=w, port=8001)

        try:
            is_ready = wait_for_service_ready("http://127.0.0.1:8001", timeout_seconds=30)
            if not is_ready:
                logger.error(f"Le serveur avec {w} workers n'a pas répondu prêt dans les délais !")
                continue

            for scenario in scenarios_list:
                for users in concurrency_list:
                    cid = f"loadtest-w{w}-{scenario}-u{users}-{uuid4().hex[:6]}"
                    tir_meta = run_locust_benchmark(
                        campaign_id=cid,
                        scenario=scenario,
                        workers_count=w,
                        users=users,
                        duration_seconds=duration_seconds,
                        output_dir=base_dir,
                    )
                    all_tirs.append(tir_meta)
                    time.sleep(2.0)  # Pause entre tirs
        finally:
            stop_uvicorn_server(server_proc)
            time.sleep(2.0)

    logger.info(f"\n[MATRIX COMPLETED] {len(all_tirs)} tirs de charge exécutés et archivés dans {base_dir}.")
    return all_tirs


def main():
    parser = argparse.ArgumentParser(description="Pilote de matrice de charge Uvicorn comparative")
    parser.add_argument("--config", type=str, default="config/loadtest.json", help="Fichier de configuration de charge")
    parser.add_argument("--quick", action="store_true", help="Mode rapide de validation (1 et 2 workers, 10s)")
    parser.add_argument("--workers", type=str, default="", help="Liste de workers séparés par virgules (ex: 1,2,4)")
    parser.add_argument("--scenarios", type=str, default="", help="Scénarios séparés par virgules (ex: single,batch,mixed)")
    parser.add_argument("--concurrency", type=str, default="", help="Paliers d'utilisateurs (ex: 5,10,20)")
    parser.add_argument("--duration", type=int, default=0, help="Durée de chaque tir en secondes")
    args = parser.parse_args()

    cfg_path = settings.PROJECT_ROOT / args.config
    cfg = {}
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

    if args.quick:
        workers = [1, 2]
        scenarios = ["single", "batch"]
        concurrency = [5, 10]
        duration = 10
    else:
        workers = [int(w.strip()) for w in args.workers.split(",") if w.strip()] if args.workers else cfg.get("workers_matrix", [1, 2, 4])
        scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()] if args.scenarios else ["single", "batch", "mixed"]
        concurrency = [int(u.strip()) for u in args.concurrency.split(",") if u.strip()] if args.concurrency else cfg.get("concurrency_matrix", [5, 10, 20])
        duration = args.duration if args.duration > 0 else cfg.get("stable_duration_seconds", 30)

    execute_capacity_matrix(
        workers_list=workers,
        scenarios_list=scenarios,
        concurrency_list=concurrency,
        duration_seconds=duration,
    )


if __name__ == "__main__":
    main()
