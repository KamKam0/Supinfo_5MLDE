"""src/orchestration/deployment.py - Déploiement et planification Prefect 3.

Fournit :
- La création et l'enregistrement du déploiement 'steam-batch-deployment'.
- Le planning pédagogique périodique (10 minutes par défaut) avec limite de concurrence = 1.
- Le point d'entrée flow.serve pour exécution locale sous contrôle de runner.
- Les fonctions d'administration du planning : pause, reprise, déclenchement unitaire.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import timedelta
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import UUID
from prefect.client.orchestration import get_client

from src.orchestration.tasks import steam_training_pipeline
from src.settings import settings

logger = logging.getLogger(__name__)

DEFAULT_DEPLOYMENT_NAME = "steam-batch-deployment"


def get_deployment_definition(
    name: str = DEFAULT_DEPLOYMENT_NAME,
    interval_minutes: int = 10,
    concurrency_limit: int = 1,
    parameters: Optional[Dict[str, Any]] = None,
) -> Any:
    """Construit l'objet RunnerDeployment conforme à la configuration du projet."""
    if "PREFECT_API_URL" not in os.environ:
        os.environ["PREFECT_API_URL"] = settings.PREFECT_API_URL

    default_params = {
        "register_models": True,
        "qualify": True,
        "promote_champion": True,
        "n_estimators": 5,
        "max_depth": 6,
    }
    if parameters:
        default_params.update(parameters)

    return steam_training_pipeline.to_deployment(
        name=name,
        description="Déploiement Prefect planifié pour l'entraînement automatique du catalogue Steam",
        interval=timedelta(minutes=interval_minutes),
        parameters=default_params,
        tags=["steam-mlops", "automated", "batch-training"],
    )


def apply_deployment(
    name: str = DEFAULT_DEPLOYMENT_NAME,
    interval_minutes: int = 10,
) -> str:
    """Enregistre le déploiement sur le serveur Prefect et retourne son ID."""
    dep = get_deployment_definition(name=name, interval_minutes=interval_minutes)
    dep_id = dep.apply()
    logger.info(f"[DEPLOYMENT] Déploiement '{name}' enregistré sur Prefect Server avec ID : {dep_id}")
    return str(dep_id)


async def _get_deployment_by_name(client: Any, name: str) -> Optional[Any]:
    deployments = await client.read_deployments()
    for d in deployments:
        if d.name == name:
            return d
    return None


async def pause_deployment_schedule(name: str = DEFAULT_DEPLOYMENT_NAME) -> bool:
    """Met en pause le planning automatique d'un déploiement."""
    if "PREFECT_API_URL" not in os.environ:
        os.environ["PREFECT_API_URL"] = settings.PREFECT_API_URL

    async with get_client() as client:
        dep = await _get_deployment_by_name(client, name)
        if not dep:
            logger.warning(f"[DEPLOYMENT] Déploiement '{name}' introuvable.")
            return False
        await client.pause_deployment(dep.id)
        logger.info(f"[DEPLOYMENT] Planning du déploiement '{name}' ({dep.id}) mis en PAUSE.")
        return True


async def resume_deployment_schedule(name: str = DEFAULT_DEPLOYMENT_NAME) -> bool:
    """Réactive le planning automatique d'un déploiement."""
    if "PREFECT_API_URL" not in os.environ:
        os.environ["PREFECT_API_URL"] = settings.PREFECT_API_URL

    async with get_client() as client:
        dep = await _get_deployment_by_name(client, name)
        if not dep:
            logger.warning(f"[DEPLOYMENT] Déploiement '{name}' introuvable.")
            return False
        await client.resume_deployment(dep.id)
        logger.info(f"[DEPLOYMENT] Planning du déploiement '{name}' ({dep.id}) RÉACTIVÉ.")
        return True


async def trigger_deployment_run(
    name: str = DEFAULT_DEPLOYMENT_NAME,
    parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Déclenche immédiatement une exécution du déploiement depuis l'API Prefect."""
    if "PREFECT_API_URL" not in os.environ:
        os.environ["PREFECT_API_URL"] = settings.PREFECT_API_URL

    async with get_client() as client:
        dep = await _get_deployment_by_name(client, name)
        if not dep:
            raise ValueError(f"Déploiement '{name}' introuvable sur le serveur Prefect.")

        flow_run = await client.create_flow_run_from_deployment(
            deployment_id=dep.id,
            parameters=parameters or {},
            state=None,
        )
        logger.info(
            f"[DEPLOYMENT] Flow run déclenché avec succès depuis le déploiement '{name}' : "
            f"id={flow_run.id}, name={flow_run.name}"
        )
        return {
            "deployment_name": name,
            "deployment_id": str(dep.id),
            "flow_run_id": str(flow_run.id),
            "flow_run_name": flow_run.name,
            "state": flow_run.state.name if flow_run.state else "Scheduled",
        }


def serve_deployment(
    name: str = DEFAULT_DEPLOYMENT_NAME,
    interval_minutes: int = 10,
    limit: int = 1,
) -> None:
    """Lance le processus local de service du déploiement via flow.serve."""
    if "PREFECT_API_URL" not in os.environ:
        os.environ["PREFECT_API_URL"] = settings.PREFECT_API_URL

    print(
        f"[DEPLOYMENT] Démarrage du runner local flow.serve pour '{name}' "
        f"(planning: {interval_minutes} min, limite concurrence: {limit})..."
    )
    steam_training_pipeline.serve(
        name=name,
        interval=timedelta(minutes=interval_minutes),
        limit=limit,
        tags=["steam-mlops", "local-runner"],
        parameters={"n_estimators": 5, "max_depth": 6},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Gestionnaire de déploiement Prefect pour Steam MLOps")
    parser.add_argument("--deploy", action="store_true", help="Enregistrer le déploiement sur le serveur Prefect")
    parser.add_argument("--serve", action="store_true", help="Exécuter le service de déploiement en local (flow.serve)")
    parser.add_argument("--run-once", action="store_true", help="Déclencher une exécution du déploiement existant")
    parser.add_argument("--pause", action="store_true", help="Mettre en pause le planning du déploiement")
    parser.add_argument("--resume", action="store_true", help="Réactiver le planning du déploiement")
    parser.add_argument("--interval", type=int, default=10, help="Intervalle de planification en minutes")
    args = parser.parse_args()

    if "PREFECT_API_URL" not in os.environ:
        os.environ["PREFECT_API_URL"] = settings.PREFECT_API_URL

    if args.deploy:
        dep_id = apply_deployment(interval_minutes=args.interval)
        print(f"[OK] Déploiement appliqué avec succès : ID = {dep_id}")
    elif args.run_once:
        run_res = asyncio.run(trigger_deployment_run())
        print(f"[OK] Run déclenché : {run_res}")
    elif args.pause:
        res = asyncio.run(pause_deployment_schedule())
        print(f"[OK] Mise en pause : {res}")
    elif args.resume:
        res = asyncio.run(resume_deployment_schedule())
        print(f"[OK] Reprise : {res}")
    elif args.serve:
        serve_deployment(interval_minutes=args.interval)
    else:
        dep_id = apply_deployment(interval_minutes=args.interval)
        print(f"[OK] Déploiement appliqué par défaut : ID = {dep_id}")


if __name__ == "__main__":
    main()
