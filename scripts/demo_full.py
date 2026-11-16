#!/usr/bin/env python3
"""scripts/demo_full.py - Démonstrateur intégré complet (Nominal + Incidents + Rollback).

Wrapper unifié appelant demo_nominal.py avec l'option --with-incidents pour couvrir
l'ensemble du cycle de vie opérationnel du projet :
1. Parcours nominal complet (7 étapes : Doctor, GX, Prefect, MLflow, Inférence, Feedback, Audit SQL).
2. Démonstrateur d'incidents (5 scénarios : GX invalide, crash PostgreSQL 503, MLflow reconcile, Rollback d'urgence, Alert down).

Usage :
    uv run --locked python scripts/demo_full.py
    uv run --locked python scripts/demo_full.py --campaign soutenance-finale
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.demo_nominal import run_nominal_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Démonstrateur intégré complet MLOps 5MLDE")
    parser.add_argument("--campaign", type=str, default="demo-full", help="Nom de la campagne")
    parser.add_argument("--fast", action="store_true", default=True, help="Mode rapide pour l'entraînement")
    parser.add_argument("--full-training", action="store_false", dest="fast", help="Entraînement complet")
    args = parser.parse_args()

    run_nominal_demo(
        campaign=args.campaign,
        fast=args.fast,
        with_incidents=True,
    )


if __name__ == "__main__":
    main()
