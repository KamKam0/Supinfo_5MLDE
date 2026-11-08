"""tests/contracts/test_data_contracts.py - Tests de validation des contrats de données Pydantic."""

import pytest
from pydantic import ValidationError
from uuid import uuid4

from src.contracts import (
    FeedbackInput,
    HistoricalGameRow,
    InferenceBatchRequest,
    InferenceInputGame,
    NewObservationGameRow,
)


def test_inference_input_valid():
    """Valide qu'une entrée d'inférence contenant les features autorisées est acceptée."""
    game = InferenceInputGame(
        appid=730,
        name="Counter-Strike 2",
        price=0.0,
        required_age=0,
        dlc_count=1,
        achievements=1,
        windows=True,
        mac=False,
        linux=True,
        genres="['Action', 'Free To Play']",
        release_date="2012-08-21",
    )
    assert game.appid == 730
    assert game.price == 0.0


def test_inference_input_rejects_target_pct_pos_total():
    """Vérifie le rejet formel de la cible pct_pos_total dans l'entrée d'inférence (anti-fuite)."""
    with pytest.raises(ValidationError) as exc:
        InferenceInputGame(
            appid=730,
            price=0.0,
            pct_pos_total=85.0,  # Variable cible interdite
        )
    assert "extra_forbidden" in str(exc.value)


def test_inference_input_rejects_target_estimated_owners():
    """Vérifie le rejet formel de la cible estimated_owners dans l'entrée d'inférence (anti-fuite)."""
    with pytest.raises(ValidationError) as exc:
        InferenceInputGame(
            appid=730,
            price=0.0,
            estimated_owners="20000 - 50000",  # Variable cible interdite
        )
    assert "extra_forbidden" in str(exc.value)


def test_inference_input_rejects_leakage_review_counts():
    """Vérifie le rejet formel des colonnes fuyantes (positive, negative, num_reviews_total)."""
    with pytest.raises(ValidationError):
        InferenceInputGame(appid=730, positive=100)
    with pytest.raises(ValidationError):
        InferenceInputGame(appid=730, negative=10)
    with pytest.raises(ValidationError):
        InferenceInputGame(appid=730, num_reviews_total=110)


def test_inference_batch_bounds():
    """Vérifie les bornes de taille d'un batch de prédiction (1 à 100)."""
    item = InferenceInputGame(appid=730, price=0.0)

    # Batch vide interdit
    with pytest.raises(ValidationError):
        InferenceBatchRequest(items=[])

    # Batch de 1 valide
    batch_valid = InferenceBatchRequest(items=[item])
    assert len(batch_valid.items) == 1

    # Batch > 100 interdit
    with pytest.raises(ValidationError):
        InferenceBatchRequest(items=[item] * 101)


def test_historical_vs_new_observation_sentinels():
    """Vérifie la tolérance des sentinelles sur l'historique et leur rejet sur les nouvelles données."""
    # L'historique accepte required_age = -1 et pct_pos_total = -1
    hist = HistoricalGameRow(appid=123, required_age=-1, pct_pos_total=-1.0)
    assert hist.required_age == -1

    # Une nouvelle observation rejette required_age = -1
    with pytest.raises(ValidationError):
        NewObservationGameRow(appid=123, price=9.99, required_age=-1)


def test_feedback_input_validation():
    """Vérifie la cohérence métier et SQL des feedbacks utilisateurs."""
    req_id = uuid4()
    pred_id = uuid4()

    # Feedback qualitatif valide
    fb_qual = FeedbackInput(
        request_id=req_id,
        prediction_id=pred_id,
        task="regression",
        useful=True,
    )
    assert fb_qual.useful is True

    # Feedback régression avec vérité terrain valide
    fb_reg = FeedbackInput(
        request_id=req_id,
        prediction_id=pred_id,
        task="regression",
        observed_pct=88.5,
    )
    assert fb_reg.observed_pct == 88.5

    # Erreur si feedback sans aucun indicateur (ni useful, ni observed_*)
    with pytest.raises(ValidationError):
        FeedbackInput(
            request_id=req_id,
            prediction_id=pred_id,
            task="regression",
        )

    # Erreur si format classification envoyé pour régression
    with pytest.raises(ValidationError):
        FeedbackInput(
            request_id=req_id,
            prediction_id=pred_id,
            task="regression",
            observed_owners="20000 - 50000",
        )
