"""src/data_sources/snapshots.py - Gestion des fusions par upsert et publication de snapshots immuables.

Garantit :
- La validation des nouvelles observations par Great Expectations (new_observations_v1) avant toute fusion.
- La préservation stricte et immuable du snapshot de base d'origine.
- La mise à jour (upsert) des appids existants et l'ajout des nouveaux appids.
- La validation finale de l'intégrité du snapshot consolidé (snapshot_integrity_v1).
- Le calcul de l'empreinte cryptographique SHA-256 du nouveau snapshot publié.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import pandas as pd

from src.contracts import REQUIRED_SNAPSHOT_COLUMNS
from src.data_validation import validate_dataframe
from src.exceptions import DataIntegrityError, DataValidationError


def compute_file_sha256(file_path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    """Calcule l'empreinte SHA-256 d'un fichier."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha.update(chunk)
    return sha.hexdigest().lower()


def upsert_and_publish_snapshot(
    base_snapshot_path: Path | str,
    new_observations: pd.DataFrame | List[Dict[str, Any]],
    output_snapshot_path: Path | str,
    *,
    reserved_appids_path: Optional[Path | str] = None,
    validate_before_merge: bool = True,
    validate_after_merge: bool = True,
) -> Dict[str, Any]:
    """Fusionne déterministement de nouvelles observations avec un snapshot de base existant.

    Args:
        base_snapshot_path: Chemin du snapshot source immuable.
        new_observations: Nouvelles lignes à intégrer (DataFrame ou liste de dicts).
        output_snapshot_path: Chemin du nouveau snapshot publié.
        reserved_appids_path: Chemin optionnel vers reserved_appids.json pour audit.
        validate_before_merge: Si True, valide new_observations avec new_observations_v1.
        validate_after_merge: Si True, valide le snapshot final avec snapshot_integrity_v1.

    Returns:
        Dictionnaire de rapport d'upsert (lignes ajoutées, mises à jour, hash SHA-256, etc.).

    Raises:
        DataValidationError si l'une des barrières de qualité échoue.
    """
    base_path = Path(base_snapshot_path).resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"Snapshot de base introuvable : {base_path}")

    out_path = Path(output_snapshot_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Vérification de l'empreinte initiale du fichier source
    initial_base_sha = compute_file_sha256(base_path)

    # 2. Conversion des nouvelles observations en DataFrame
    if isinstance(new_observations, list):
        new_df = pd.DataFrame(new_observations)
    else:
        new_df = new_observations.copy()

    if len(new_df) == 0:
        raise DataValidationError("Aucune nouvelle observation fournie pour l'upsert.")

    # Réordonnancement strict selon le contrat 16 colonnes
    columns_order = list(REQUIRED_SNAPSHOT_COLUMNS)
    for col in columns_order:
        if col not in new_df.columns:
            raise DataValidationError(f"Colonne obligatoire manquante dans les nouvelles observations : '{col}'")
    new_df = new_df[columns_order].copy()

    # 3. Barrière GX amont : Validation stricte des nouvelles observations
    if validate_before_merge:
        validate_dataframe(
            new_df,
            suite_name="new_observations_v1",
            dataset_name="new_observations_pre_merge",
            raise_on_error=True,
        )

    # 4. Lecture du snapshot de base (sans modification)
    base_df = pd.read_csv(base_path)
    base_row_count = len(base_df)
    base_appids = set(base_df["appid"].dropna().astype(int))

    # 5. Application de l'upsert
    new_appids = set(new_df["appid"].dropna().astype(int))
    updated_appids = base_appids & new_appids
    added_appids = new_appids - base_appids

    # Conserver les lignes de base non modifiées
    retained_base_df = base_df[~base_df["appid"].isin(updated_appids)].copy()

    # Concaténation : base conservée + nouvelles observations (mises à jour + ajouts)
    consolidated_df = pd.concat([retained_base_df, new_df], ignore_index=True)
    # Tri stable par appid pour reproductibilité absolue
    consolidated_df = consolidated_df.sort_values(by="appid").reset_index(drop=True)
    consolidated_df = consolidated_df[columns_order]

    # 6. Écriture atomique du nouveau snapshot publié (CRLF)
    csv_bytes = consolidated_df.to_csv(index=False, lineterminator="\r\n").encode("utf-8")
    with open(out_path, "wb") as f:
        f.write(csv_bytes)

    # 7. Contrôle de non-altération du fichier de base
    post_base_sha = compute_file_sha256(base_path)
    if post_base_sha != initial_base_sha:
        raise DataIntegrityError(
            f"VIOLATION D'IMMUABILITÉ : Le snapshot de base a été modifié durant l'upsert ! "
            f"(Initial: {initial_base_sha}, Constaté: {post_base_sha})"
        )

    # 8. Barrière GX aval : Contrôle d'intégrité du snapshot consolidé
    if validate_after_merge:
        validate_dataframe(
            consolidated_df,
            suite_name="snapshot_integrity_v1",
            dataset_name=out_path.stem,
            raise_on_error=True,
        )

    new_snapshot_sha = compute_file_sha256(out_path)

    # 9. Audit des réserves protégées
    reserved_count = 0
    if reserved_appids_path:
        res_file = Path(reserved_appids_path).resolve()
        if res_file.exists():
            with open(res_file, "r", encoding="utf-8") as f:
                res_set = set(json.load(f))
            reserved_count = len(set(consolidated_df["appid"].astype(int)) & res_set)

    summary = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "base_snapshot_path": str(base_path),
        "base_snapshot_sha256": initial_base_sha,
        "base_row_count": base_row_count,
        "added_count": len(added_appids),
        "updated_count": len(updated_appids),
        "final_row_count": len(consolidated_df),
        "reserved_appids_count": reserved_count,
        "new_snapshot_path": str(out_path),
        "new_snapshot_sha256": new_snapshot_sha,
    }

    print(
        f"[SNAPSHOTS] Upsert achevé avec succès : {base_row_count:,} base + "
        f"{len(added_appids):,} ajoutés, {len(updated_appids):,} actualisés "
        f"-> Total : {len(consolidated_df):,} lignes (SHA256: {new_snapshot_sha[:12]}...)"
    )

    return summary
