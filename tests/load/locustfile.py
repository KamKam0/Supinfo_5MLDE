"""tests/load/locustfile.py - Scénarios de test de charge Locust 2.46.6.

Implémente trois profils d'utilisateurs virtuels :
1. SinglePredictionUser : Inférence unitaire (/predict).
2. BatchPredictionUser : Inférence par lot fixe de 20 jeux (/predict/batch).
3. MixedWorkloadUser : Charge mixte représentative (70% unitaire, 20% batch, 10% feedbacks réels).

Validation stricte des réponses via catch_response :
- Contrôle des statuts HTTP, détection et rejet immédiat des faux 200.
- Vérification du respect des bornes, présences des 2 tâches et intégrité structurelle.
- En-tête obligatoire de traçabilité 'X-Test-Campaign'.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from locust import HttpUser, between, task

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from tests.load.locust_helpers import (
    BATCH_20_GAMES,
    LOAD_GAMES,
    get_campaign_header,
    validate_single_prediction_response,
)


class SinglePredictionUser(HttpUser):
    """Utilisateur virtuel exécutant exclusivement des requêtes d'inférence unitaire."""

    wait_time = between(0.05, 0.2)
    weight = 1

    @task
    def predict_single(self):
        game = random.choice(LOAD_GAMES)
        headers = get_campaign_header()
        with self.client.post(
            "/predict",
            json={"game": game},
            headers=headers,
            name="/predict",
            catch_response=True,
        ) as response:
            validate_single_prediction_response(response, expected_item_count=1)


class BatchPredictionUser(HttpUser):
    """Utilisateur virtuel exécutant exclusivement des requêtes par lot fixe de 20 jeux."""

    wait_time = between(0.1, 0.5)
    weight = 1

    @task
    def predict_batch(self):
        headers = get_campaign_header()
        with self.client.post(
            "/predict/batch",
            json={"games": BATCH_20_GAMES},
            headers=headers,
            name="/predict/batch",
            catch_response=True,
        ) as response:
            validate_single_prediction_response(response, expected_item_count=20)


class MixedWorkloadUser(HttpUser):
    """Utilisateur virtuel simulant la production (70% unitaire, 20% lot fixe 20, 10% feedback)."""

    wait_time = between(0.05, 0.25)
    weight = 3

    def on_start(self):
        self.cached_prediction_ids: List[str] = []

    @task(70)
    def task_single_predict(self):
        game = random.choice(LOAD_GAMES)
        headers = get_campaign_header()
        with self.client.post(
            "/predict",
            json={"game": game},
            headers=headers,
            name="/predict",
            catch_response=True,
        ) as response:
            pids = validate_single_prediction_response(response, expected_item_count=1)
            if pids:
                # Conserve un identifiant pour le tir de feedback ultérieur
                self.cached_prediction_ids.extend(pids[:2])
                if len(self.cached_prediction_ids) > 100:
                    self.cached_prediction_ids = self.cached_prediction_ids[-50:]

    @task(20)
    def task_batch_predict(self):
        headers = get_campaign_header()
        with self.client.post(
            "/predict/batch",
            json={"games": BATCH_20_GAMES},
            headers=headers,
            name="/predict/batch",
            catch_response=True,
        ) as response:
            pids = validate_single_prediction_response(response, expected_item_count=20)
            if pids:
                self.cached_prediction_ids.extend(pids[:4])

    @task(10)
    def task_feedback(self):
        if not self.cached_prediction_ids:
            # Si aucun ID en cache, repli sur prédiction unitaire
            self.task_single_predict()
            return

        pred_id = self.cached_prediction_ids.pop(0)
        feedback_id = str(uuid4())
        payload = {
            "feedback_id": feedback_id,
            "prediction_id": pred_id,
            "useful": random.choice([True, False]),
        }
        headers = get_campaign_header()
        with self.client.post(
            "/feedback",
            json=payload,
            headers=headers,
            name="/feedback",
            catch_response=True,
        ) as response:
            if response.status_code in (200, 201):
                try:
                    data = response.json()
                    if data.get("feedback_id") == feedback_id:
                        response.success()
                    else:
                        response.failure(f"Feedback ID incohérent dans réponse : {data}")
                except Exception as exc:
                    response.failure(f"Décodage JSON impossible sur feedback : {exc}")
            else:
                response.failure(f"Code inattendu sur /feedback : {response.status_code} - {response.text[:150]}")
