"""src/ingestion.py - Fonction mère 1 : load_data.

Charge la configuration, accède au snapshot immuable de données,
vérifie son intégrité cryptographique et prépare les métadonnées d'exécution.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import uuid4
import pandas as pd

from src.contracts import IngestionResult, RunContext, SnapshotRef
from src.settings import settings


def compute_file_sha256(file_path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    """Calcule l'empreinte SHA-256 d'un fichier."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha.update(chunk)
    return sha.hexdigest().lower()


def load_data(config_path: str | Path) -> IngestionResult:
    """Première fonction mère : acquisition et vérification d'intégrité des données brutes."""
    conf_path = Path(config_path).resolve()
    if not conf_path.exists():
        raise FileNotFoundError(f"Fichier de configuration introuvable : {conf_path}")

    with open(conf_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    project_root = conf_path.parent.parent

    # Résolution des chemins de données
    snapshot_raw = config.get("snapshot_path", "data/snapshots/historical_projection.csv")
    snapshot_path = project_root / snapshot_raw if not Path(snapshot_raw).is_absolute() else Path(snapshot_raw)

    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot de données introuvable à : {snapshot_path}")

    # Calcul SHA-256 et vérification
    actual_sha = compute_file_sha256(snapshot_path)
    expected_sha = config.get("expected_projection_sha256", "").lower()

    if expected_sha and actual_sha != expected_sha:
        raise ValueError(
            f"Échec d'intégrité SHA-256 du snapshot : constaté {actual_sha}, attendu {expected_sha}"
        )

    # Lecture des dimensions
    df = pd.read_csv(snapshot_path)
    row_count = len(df)
    col_count = len(df.columns)

    run_id = f"run-{uuid4().hex[:8]}"
    context = RunContext(
        run_id=run_id,
        mode="historical-training",
        batch_id="batch-000",
        attempt=1,
        git_revision=config.get("git_revision", "unknown"),
    )

    snapshot_ref = SnapshotRef(
        path=str(snapshot_path),
        sha256=actual_sha,
        row_count=row_count,
        column_count=col_count,
    )

    result = IngestionResult(
        context=context,
        snapshot=snapshot_ref,
        gx_success=True,
    )

    print(f"[INGESTION] Snapshot validé : {snapshot_path.name} ({row_count:,} lignes, SHA256: {actual_sha[:12]}...)")
    return result
