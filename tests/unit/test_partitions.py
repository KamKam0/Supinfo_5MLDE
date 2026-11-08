"""tests/unit/test_partitions.py - Tests unitaires de partitionnement et d'étanchéité anti-fuite."""

import json
from pathlib import Path
import pandas as pd
import pytest

from src.partitions import (
    build_stratified_partitions,
    filter_out_reserved_games,
    load_reserved_appids,
    verify_partitions_integrity,
)


@pytest.fixture
def project_root():
    return Path(__file__).resolve().parent.parent.parent


def test_frozen_partitions_files_exist(project_root):
    """Vérifie l'existence des quatre fichiers JSON de partitions."""
    partitions_dir = project_root / "data" / "partitions"
    for fname in ["train_appids.json", "val_appids.json", "test_appids.json", "reserved_appids.json"]:
        p = partitions_dir / fname
        assert p.exists(), f"Fichier de partition manquant : {p}"


def test_frozen_partitions_strictly_disjoint(project_root):
    """Vérifie mathématiquement qu'aucune intersection n'existe entre les partitions figées."""
    partitions_dir = project_root / "data" / "partitions"

    train_ids = json.loads((partitions_dir / "train_appids.json").read_text())
    val_ids = json.loads((partitions_dir / "val_appids.json").read_text())
    test_ids = json.loads((partitions_dir / "test_appids.json").read_text())
    reserved_ids = json.loads((partitions_dir / "reserved_appids.json").read_text())

    # Pas d'intersection
    assert set(train_ids).isdisjoint(set(val_ids)), "Fuite entre Train et Validation !"
    assert set(train_ids).isdisjoint(set(test_ids)), "Fuite entre Train et Test !"
    assert set(val_ids).isdisjoint(set(test_ids)), "Fuite entre Validation et Test !"

    # Cohérence de la réserve (30% = Val + Test)
    assert set(reserved_ids) == (set(val_ids) | set(test_ids))

    # Effectif total = 94 948
    total = len(train_ids) + len(val_ids) + len(test_ids)
    assert total == 94948
    assert len(train_ids) == 66463
    assert len(val_ids) == 14242
    assert len(test_ids) == 14243


def test_verify_partitions_integrity_raises_on_leakage():
    """Vérifie que la fonction de contrôle lève une exception explicite en cas d'intrusion."""
    train = [1, 2, 3]
    val = [4, 5]
    test = [3, 6]  # 3 est présent dans train et test !

    with pytest.raises(ValueError, match="FUITE DÉTECTÉE"):
        verify_partitions_integrity(train, val, test)


def test_filter_out_reserved_games():
    """Vérifie que tout lot entrant contenant un appid réservé est automatiquement expurgé."""
    reserved = {100, 200, 300}
    candidates = pd.DataFrame([
        {"appid": 50, "price": 10.0},
        {"appid": 100, "price": 15.0},  # Réservé !
        {"appid": 150, "price": 20.0},
        {"appid": 300, "price": 25.0},  # Réservé !
    ])

    filtered = filter_out_reserved_games(candidates, reserved)
    assert len(filtered) == 2
    assert set(filtered["appid"]) == {50, 150}
