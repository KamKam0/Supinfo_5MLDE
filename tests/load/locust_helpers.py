"""tests/load/locust_helpers.py - Fonctions utilitaires et validation pour les tests de charge Locust.

Permet de découpler la logique de validation et le chargement des fixtures de Locust,
évitant les collisions de monkey-patching gevent/ssl lors des suites pytest complètes.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Chargement des fixtures étalonnées
FIXTURES_PATH = PROJECT_ROOT / "fixtures" / "load_fixtures.json"
if FIXTURES_PATH.exists():
    with open(FIXTURES_PATH, "r", encoding="utf-8") as f:
        _fixtures_data = json.load(f)
        LOAD_GAMES: List[Dict[str, Any]] = _fixtures_data.get("games", [])
else:
    LOAD_GAMES = [
        {
            "appid": 1086940,
            "name": "Baldur's Gate 3",
            "price": 59.99,
            "required_age": 18,
            "genres": ["RPG", "Strategy", "Adventure"],
            "windows": True,
            "mac": True,
            "linux": False,
            "release_date": "2023-08-03",
        }
    ]

# Lot fixe de 20 jeux pour les scénarios batch
BATCH_20_GAMES: List[Dict[str, Any]] = LOAD_GAMES[:20] if len(LOAD_GAMES) >= 20 else (LOAD_GAMES * 20)[:20]


def get_campaign_header() -> Dict[str, str]:
    """Retourne l'en-tête de traçabilité de campagne configuré."""
    campaign_id = os.environ.get("LOCUST_CAMPAIGN_ID", "loadtest-default")
    return {"X-Test-Campaign": campaign_id[:128]}


def validate_single_prediction_response(response: Any, expected_item_count: int = 1) -> Optional[List[str]]:
    """Valide rigoureusement le corps d'une réponse de prédiction unitaire ou lot.

    Détecte et rejette comme échec tout code 200 corrompu ou incomplet.
    Retourne la liste des prediction_ids extraits en cas de succès.
    """
    if response.status_code != 200:
        response.failure(f"Code HTTP inattendu : {response.status_code} (attendu: 200) - {response.text[:200]}")
        return None

    try:
        data = response.json()
    except Exception as exc:
        response.failure(f"Échec de décodage JSON sur code 200 : {exc}")
        return None

    # Contrôles de structure
    if "request_id" not in data or "results" not in data or "item_count" not in data:
        response.failure("Structure JSON incomplète : champs obligatoires manquants")
        return None

    if data["item_count"] != expected_item_count:
        response.failure(f"item_count incohérent : {data['item_count']} (attendu: {expected_item_count})")
        return None

    results = data["results"]
    if len(results) != expected_item_count:
        response.failure(f"Nombre de résultats incorrect : {len(results)} (attendu: {expected_item_count})")
        return None

    extracted_prediction_ids = []

    # Vérification unitaire de chaque résultat
    for i, item in enumerate(results):
        if item.get("item_index") != i:
            response.failure(f"Index séquentiel erroné au rang {i} : {item.get('item_index')}")
            return None

        # Vérification des prédictions conjointes (régression + classification)
        preds = item.get("predictions", [])
        if len(preds) != 2:
            response.failure(f"Résultat {i} n'a pas exactement 2 prédictions (trouvé: {len(preds)})")
            return None

        reg_pred = next((p for p in preds if p.get("task") == "regression"), None)
        cls_pred = next((p for p in preds if p.get("task") == "classification"), None)

        if not reg_pred or not cls_pred:
            response.failure(f"Résultat {i} ne contient pas les deux tâches (régression et classification)")
            return None

        # Vérification des bornes
        pct = item.get("predicted_pct")
        if pct is None or not (0.0 <= pct <= 100.0):
            response.failure(f"Pourcentage prédit hors bornes [0, 100] : {pct}")
            return None

        owners = item.get("predicted_owners")
        if not owners or " - " not in owners:
            response.failure(f"Format de tranche propriétaires invalide : {owners}")
            return None

        extracted_prediction_ids.append(reg_pred["prediction_id"])
        extracted_prediction_ids.append(cls_pred["prediction_id"])

    response.success()
    return extracted_prediction_ids
