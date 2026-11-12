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
from src.data_validation import validate_dataframe
from src.settings import settings


def compute_file_sha256(file_path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    """Calcule l'empreinte SHA-256 d'un fichier."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha.update(chunk)
    return sha.hexdigest().lower()


def load_data(config_input: str | Path | dict) -> IngestionResult:
    """Première fonction mère : acquisition et vérification d'intégrité des données brutes."""
    if isinstance(config_input, dict):
        config = config_input
        project_root = Path(__file__).resolve().parent.parent
    else:
        conf_path = Path(config_input).resolve()
        if not conf_path.exists():
            raise FileNotFoundError(f"Fichier de configuration introuvable : {conf_path}")
        with open(conf_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        project_root = conf_path.parent.parent

    # Résolution des chemins de données
    snapshot_raw = config.get("snapshot_path", "data/snapshots/historical_projection.csv")
    snapshot_path = project_root / snapshot_raw if not Path(snapshot_raw).is_absolute() else Path(snapshot_raw)

    # Mode Acquisition active PULL
    if config.get("mode") == "pull" or "pull_target_appids" in config:
        from src.data_sources.steam_client import SteamPULLClient
        from src.data_sources.snapshots import upsert_and_publish_snapshot

        base_raw = config.get("base_snapshot_path", "data/snapshots/historical_projection.csv")
        base_path = project_root / base_raw if not Path(base_raw).is_absolute() else Path(base_raw)
        target_appids = config.get("pull_target_appids", [620, 400, 550])

        client = SteamPULLClient()
        new_obs = client.fetch_batch(target_appids)

        upsert_res = upsert_and_publish_snapshot(
            base_snapshot_path=base_path,
            new_observations=new_obs,
            output_snapshot_path=snapshot_path,
            reserved_appids_path=project_root / "data" / "partitions" / "reserved_appids.json",
        )

    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot de données introuvable à : {snapshot_path}")

    # Calcul SHA-256 et vérification
    actual_sha = compute_file_sha256(snapshot_path)
    expected_sha = config.get("expected_projection_sha256", "").lower()

    if expected_sha and actual_sha != expected_sha:
        raise ValueError(
            f"Échec d'intégrité SHA-256 du snapshot : constaté {actual_sha}, attendu {expected_sha}"
        )

    # Lecture des dimensions et validation bloquante Great Expectations
    df = pd.read_csv(snapshot_path)
    row_count = len(df)
    col_count = len(df.columns)

    suite_name = config.get("validation_suite", "historical_schema_v1")
    validation_report = validate_dataframe(
        df,
        suite_name=suite_name,
        dataset_name=snapshot_path.stem,
        raise_on_error=True,
    )
    gx_success = bool(validation_report.get("success", False))

    mode_name = config.get("mode") or ("pull-training" if config.get("mode") == "pull" else "historical-training")
    run_id = config.get("run_id") or f"run-{uuid4().hex[:8]}"
    batch_id = config.get("batch_id") or ("batch-pull" if config.get("mode") == "pull" else "batch-000")
    flow_run_id = config.get("flow_run_id")
    attempt = int(config.get("attempt", 1))

    context = RunContext(
        run_id=run_id,
        mode=mode_name,
        batch_id=batch_id,
        attempt=attempt,
        git_revision=config.get("git_revision", "unknown"),
        flow_run_id=flow_run_id,
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
        gx_success=gx_success,
    )

    print(
        f"[INGESTION] Snapshot validé par GX ({suite_name}) : {snapshot_path.name} "
        f"({row_count:,} lignes, SHA256: {actual_sha[:12]}...)"
    )
    return result
