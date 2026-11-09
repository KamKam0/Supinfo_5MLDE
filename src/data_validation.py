"""src/data_validation.py - Barrière de qualité bloquante Great Expectations Core 1.x.

Implémente la validation bloquante des flux de données Steam :
- 5 suites : historical_schema_v1, new_observations_v1, snapshot_integrity_v1, training_readiness_v1, push_batch_v1.
- 10 familles de règles : structure, types, bornes, sentinelles, cohérence des cibles, absence de fuite.
- Rapports d'anomalies horodatés archivés dans reports/validation/.
- Levée de DataValidationError bloquant immédiatement l'exécution en cas de non-conformité.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import great_expectations as gx
import great_expectations.expectations as gxe
import numpy as np
import pandas as pd

from src.contracts import REQUIRED_SNAPSHOT_COLUMNS
from src.exceptions import (
    DataIntegrityError,
    DataLeakageError,
    DataSchemaError,
    DataValidationError,
)
from src.settings import settings


# Liste officielle des colonnes attendues dans la projection Steam 16 colonnes
HISTORICAL_COLUMNS = list(REQUIRED_SNAPSHOT_COLUMNS)

# Suites supportées
VALID_SUITES = {
    "historical_schema_v1",
    "new_observations_v1",
    "snapshot_integrity_v1",
    "training_readiness_v1",
    "push_batch_v1",
}


def build_historical_schema_suite() -> gx.ExpectationSuite:
    """Suite 1 : Projection historique entière (tolère sentinelles historiques documentées)."""
    suite = gx.ExpectationSuite(name="historical_schema_v1")

    # 1. Structure : 16 colonnes
    suite.add_expectation(
        gxe.ExpectTableColumnsToMatchSet(column_set=HISTORICAL_COLUMNS, exact_match=False)
    )
    # 2. Volume : non vide
    suite.add_expectation(gxe.ExpectTableRowCountToBeBetween(min_value=1))

    # 3. Identifiants
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="appid"))
    suite.add_expectation(gxe.ExpectColumnValuesToBeUnique(column="appid"))
    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column="appid", min_value=1))

    # 4. Compteurs et types numériques positifs
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInTypeList(
            column="price", type_list=["float64", "int64", "float", "int"]
        )
    )
    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column="price", min_value=0.0))
    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column="dlc_count", min_value=0))
    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column="achievements", min_value=0))
    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column="positive", min_value=0))
    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column="negative", min_value=0))
    # Dans l'historique brut, num_reviews_total = -1 lorsque pct_pos_total = -1 (39 575 jeux)
    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column="num_reviews_total", min_value=-1))

    # 5. Sentinelle historique autorisée pour required_age : -1 ou >= 0
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(column="required_age", min_value=-1, max_value=120)
    )

    # 6. Pourcentage d'avis positif : -1 (si reviews < 10) ou dans [0, 100]
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(column="pct_pos_total", min_value=-1.0, max_value=100.0)
    )

    # 7. Plateformes booléennes
    for plat in ["windows", "mac", "linux"]:
        suite.add_expectation(
            gxe.ExpectColumnValuesToBeInSet(column=plat, value_set=[True, False, 1, 0])
        )

    # 8. Date de sortie au format ISO YYYY-MM-DD
    suite.add_expectation(
        gxe.ExpectColumnValuesToMatchRegex(column="release_date", regex=r"^\d{4}-\d{2}-\d{2}$")
    )

    # 9. Format des tranches estimated_owners (deux entiers séparés par tiret)
    suite.add_expectation(
        gxe.ExpectColumnValuesToMatchRegex(column="estimated_owners", regex=r"^\d+ - \d+$")
    )

    return suite


def build_new_observations_suite() -> gx.ExpectationSuite:
    """Suite 2 : Nouvelles collectes (PULL ou PUSH). Aucune sentinelle négative autorisée."""
    suite = gx.ExpectationSuite(name="new_observations_v1")

    # 1. Structure
    suite.add_expectation(
        gxe.ExpectTableColumnsToMatchSet(column_set=HISTORICAL_COLUMNS, exact_match=False)
    )
    suite.add_expectation(gxe.ExpectTableRowCountToBeBetween(min_value=1))

    # 2. Identifiants
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="appid"))
    suite.add_expectation(gxe.ExpectColumnValuesToBeUnique(column="appid"))
    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column="appid", min_value=1))

    # 3. Pas de sentinelle -1 dans required_age : 0 <= required_age <= 120
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(column="required_age", min_value=0, max_value=120)
    )

    # 4. Prix et compteurs
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInTypeList(
            column="price", type_list=["float64", "int64", "float", "int"]
        )
    )
    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column="price", min_value=0.0))
    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column="dlc_count", min_value=0))
    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column="achievements", min_value=0))

    # 5. Pas de sentinelle -1 dans pct_pos_total : cibles obligatoires dans [0, 100]
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(column="pct_pos_total", min_value=0.0, max_value=100.0)
    )

    # 6. estimated_owners : regex et refus explicite de '0 - 0'
    suite.add_expectation(
        gxe.ExpectColumnValuesToMatchRegex(column="estimated_owners", regex=r"^\d+ - \d+$")
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToNotBeInSet(
            column="estimated_owners", value_set=[settings.CLASSIFICATION_EXCLUDED_CLASS]
        )
    )

    # 7. Plateformes booléennes
    for plat in ["windows", "mac", "linux"]:
        suite.add_expectation(
            gxe.ExpectColumnValuesToBeInSet(column=plat, value_set=[True, False, 1, 0])
        )

    # 8. Date ISO
    suite.add_expectation(
        gxe.ExpectColumnValuesToMatchRegex(column="release_date", regex=r"^\d{4}-\d{2}-\d{2}$")
    )

    return suite


def build_snapshot_integrity_suite() -> gx.ExpectationSuite:
    """Suite 3 : Intégrité d'un snapshot candidat complet (unicité, volume, non-nullité clé)."""
    suite = gx.ExpectationSuite(name="snapshot_integrity_v1")

    suite.add_expectation(
        gxe.ExpectTableColumnsToMatchSet(column_set=HISTORICAL_COLUMNS, exact_match=False)
    )
    suite.add_expectation(gxe.ExpectTableRowCountToBeBetween(min_value=1000))
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="appid"))
    suite.add_expectation(gxe.ExpectColumnValuesToBeUnique(column="appid"))
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="name"))
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="release_date"))
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="price"))
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="pct_pos_total"))
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="estimated_owners"))

    return suite


def build_training_readiness_suite() -> gx.ExpectationSuite:
    """Suite 4 : Vérification d'aptitude à l'entraînement (populations éligibles, diversité de classes)."""
    suite = gx.ExpectationSuite(name="training_readiness_v1")

    suite.add_expectation(gxe.ExpectTableRowCountToBeBetween(min_value=10))
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="appid"))
    suite.add_expectation(gxe.ExpectColumnValuesToBeUnique(column="appid"))
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="pct_pos_total"))
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="estimated_owners"))

    return suite


def build_push_batch_suite() -> gx.ExpectationSuite:
    """Suite 5 : Validation d'un lot PUSH figé de 500 jeux uniques."""
    suite = gx.ExpectationSuite(name="push_batch_v1")

    # Exactement 500 jeux
    suite.add_expectation(gxe.ExpectTableRowCountToEqual(value=500))
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="appid"))
    suite.add_expectation(gxe.ExpectColumnValuesToBeUnique(column="appid"))
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(column="pct_pos_total", min_value=0.0, max_value=100.0)
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToMatchRegex(column="estimated_owners", regex=r"^\d+ - \d+$")
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToNotBeInSet(
            column="estimated_owners", value_set=[settings.CLASSIFICATION_EXCLUDED_CLASS]
        )
    )

    return suite


def get_suite_by_name(suite_name: str) -> gx.ExpectationSuite:
    """Retourne l'objet ExpectationSuite correspondant au nom donné."""
    builders = {
        "historical_schema_v1": build_historical_schema_suite,
        "new_observations_v1": build_new_observations_suite,
        "snapshot_integrity_v1": build_snapshot_integrity_suite,
        "training_readiness_v1": build_training_readiness_suite,
        "push_batch_v1": build_push_batch_suite,
    }
    if suite_name not in builders:
        raise ValueError(
            f"Nom de suite inconnu : '{suite_name}'. Suites valides : {sorted(VALID_SUITES)}"
        )
    return builders[suite_name]()


def _json_default_serializer(obj: Any) -> Any:
    """Sérialiseur de secours pour types numpy, sets et chemins."""
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.ndarray, set)):
        return list(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def validate_dataframe(
    df: pd.DataFrame,
    suite_name: str,
    *,
    dataset_name: str = "dataset",
    check_reserved: bool = False,
    reserved_appids: Optional[Set[int]] = None,
    raise_on_error: bool = True,
    output_dir: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Exécute la barrière de validation Great Expectations Core 1.x et les contrôles métier.

    Args:
        df: Le DataFrame pandas à valider.
        suite_name: Nom de la suite GX ('historical_schema_v1', 'new_observations_v1', etc.).
        dataset_name: Nom informatif pour le rapport d'audit.
        check_reserved: Si True, vérifie l'absence absolue d'appids réservés.
        reserved_appids: Ensemble des appids réservés à exclure (si check_reserved=True).
        raise_on_error: Si True, lève DataValidationError lors d'un échec.
        output_dir: Répertoire de destination des rapports de validation.

    Returns:
        Dictionnaire récapitulatif du rapport de validation.

    Raises:
        DataValidationError (ou sous-classe) si validation négative et raise_on_error=True.
    """
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report_dir = Path(output_dir).resolve() if output_dir else Path("reports/validation").resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    failed_checks: List[Dict[str, Any]] = []
    offending_appids: Set[Any] = set()

    # --------------------------------------------------------------------------
    # 1. Barrière Great Expectations Core 1.x (Contexte Éphémère)
    # --------------------------------------------------------------------------
    try:
        context = gx.get_context(mode="ephemeral")
        data_source = context.data_sources.add_pandas("validation_datasource")
        data_asset = data_source.add_dataframe_asset(f"{dataset_name}_asset")
        batch_def = data_asset.add_batch_definition_whole_dataframe(f"{dataset_name}_batch")

        suite = get_suite_by_name(suite_name)
        context.suites.add(suite)

        val_def = gx.ValidationDefinition(
            name=f"{suite_name}_validation", data=batch_def, suite=suite
        )
        context.validation_definitions.add(val_def)

        gx_result = val_def.run(batch_parameters={"dataframe": df})
        gx_success = bool(gx_result.success)

        # Collecte des anomalies détectées par GX
        for res in gx_result.results:
            if not res.success:
                exp_type = getattr(res.expectation_config, "type", "unknown_expectation")
                kwargs = getattr(res.expectation_config, "kwargs", {})
                res_dict = getattr(res, "result", {}) or {}

                unexpected_indices = res_dict.get("partial_unexpected_index_list", [])
                if unexpected_indices and "appid" in df.columns:
                    sample_appids = df.loc[unexpected_indices[:10], "appid"].tolist()
                    offending_appids.update(sample_appids)

                failed_checks.append(
                    {
                        "source": "great_expectations",
                        "expectation": exp_type,
                        "column": kwargs.get("column", "table"),
                        "details": res_dict,
                    }
                )

    except Exception as exc:
        failed_checks.append(
            {
                "source": "gx_execution_exception",
                "error": str(exc),
            }
        )
        gx_success = False

    # --------------------------------------------------------------------------
    # 2. Contrôles Métier Complémentaires (Ordre des bornes de tranches, etc.)
    # --------------------------------------------------------------------------
    # Contrôle de l'ordre strictement croissant des tranches d'owners
    if "estimated_owners" in df.columns:
        for idx, val in df["estimated_owners"].dropna().items():
            val_str = str(val).strip()
            if re.match(r"^\d+ - \d+$", val_str):
                parts = val_str.split(" - ")
                try:
                    low, high = int(parts[0]), int(parts[1])
                    if suite_name == "historical_schema_v1":
                        # Autorise '0 - 0' dans l'historique brut
                        if low > high:
                            failed_checks.append(
                                {
                                    "source": "business_rule",
                                    "rule": "inverted_owner_bounds",
                                    "row_index": int(idx),
                                    "value": val_str,
                                }
                            )
                            if "appid" in df.columns:
                                offending_appids.add(int(df.at[idx, "appid"]))
                    else:
                        # Refuse toute borne inversée ou égale (dont '0 - 0') dans les nouvelles collectes
                        if low >= high:
                            failed_checks.append(
                                {
                                    "source": "business_rule",
                                    "rule": "invalid_or_inverted_owner_bounds",
                                    "row_index": int(idx),
                                    "value": val_str,
                                }
                            )
                            if "appid" in df.columns:
                                offending_appids.add(int(df.at[idx, "appid"]))
                except ValueError:
                    pass

    # Contrôle d'aptitude à l'entraînement (training_readiness_v1)
    if suite_name == "training_readiness_v1":
        # Multi-classe : au moins 2 classes avec effectif suffisant
        if "estimated_owners" in df.columns:
            cls_valid = df[df["estimated_owners"] != settings.CLASSIFICATION_EXCLUDED_CLASS]
            class_counts = cls_valid["estimated_owners"].value_counts()
            active_classes = class_counts[
                class_counts >= settings.CLASSIFICATION_MIN_CLASS_COUNT
            ]
            if len(active_classes) < 2:
                failed_checks.append(
                    {
                        "source": "readiness_rule",
                        "rule": "insufficient_classes_for_stratified_training",
                        "active_classes_count": len(active_classes),
                        "required_min_classes": 2,
                    }
                )

        # Régression : effectif éligible non nul
        if "pct_pos_total" in df.columns and "num_reviews_total" in df.columns:
            reg_count = (
                (df["pct_pos_total"] >= settings.REGRESSION_MIN_PCT)
                & (df["pct_pos_total"] <= settings.REGRESSION_MAX_PCT)
                & (df["num_reviews_total"] >= settings.REGRESSION_MIN_REVIEWS)
            ).sum()
            if reg_count == 0:
                failed_checks.append(
                    {
                        "source": "readiness_rule",
                        "rule": "empty_regression_eligible_population",
                    }
                )

    # Règle anti-fuite : absence totale d'appids réservés
    if check_reserved and reserved_appids and "appid" in df.columns:
        present_appids = set(df["appid"].dropna().astype(int))
        leaked = present_appids & reserved_appids
        if leaked:
            offending_appids.update(leaked)
            failed_checks.append(
                {
                    "source": "leakage_rule",
                    "rule": "reserved_appids_found_in_training_flow",
                    "leaked_count": len(leaked),
                    "sample_leaked_appids": sorted(leaked)[:10],
                }
            )

    # --------------------------------------------------------------------------
    # 3. Établissement du Bilan & Archivage
    # --------------------------------------------------------------------------
    overall_success = (len(failed_checks) == 0) and gx_success

    report_summary: Dict[str, Any] = {
        "timestamp": timestamp,
        "suite_name": suite_name,
        "dataset_name": dataset_name,
        "row_count": len(df),
        "column_count": len(df.columns),
        "success": overall_success,
        "failed_checks_count": len(failed_checks),
        "failed_checks": failed_checks,
        "sample_offending_appids": sorted(list(offending_appids))[:20],
    }

    # Sauvegarde du rapport "latest"
    latest_file = report_dir / "latest_validation_report.json"
    with open(latest_file, "w", encoding="utf-8") as f:
        json.dump(report_summary, f, indent=2, ensure_ascii=False, default=_json_default_serializer)

    # Si échec, archivage permanent avec horodatage
    if not overall_success:
        clean_ts = timestamp.replace(":", "-").replace(".", "_")
        fail_file = report_dir / f"validation_failure_{suite_name}_{clean_ts}.json"
        with open(fail_file, "w", encoding="utf-8") as f:
            json.dump(report_summary, f, indent=2, ensure_ascii=False, default=_json_default_serializer)
        report_summary["archived_report_path"] = str(fail_file)

        if raise_on_error:
            # Sélection de la sous-classe appropriée
            if any(c.get("source") == "leakage_rule" for c in failed_checks):
                raise DataLeakageError(
                    f"[QUALITÉ DONNÉES] Rejet bloquant : détection d'appids réservés dans le flux ({len(offending_appids)} détectés).",
                    suite_name=suite_name,
                    failed_expectations=failed_checks,
                    sample_offending_appids=report_summary["sample_offending_appids"],
                    report_path=str(fail_file),
                )
            if any("column" in c and c["column"] == "table" for c in failed_checks):
                raise DataSchemaError(
                    f"[QUALITÉ DONNÉES] Rejet bloquant : schéma invalide ou colonnes manquantes.",
                    suite_name=suite_name,
                    failed_expectations=failed_checks,
                    sample_offending_appids=report_summary["sample_offending_appids"],
                    report_path=str(fail_file),
                )
            raise DataValidationError(
                f"[QUALITÉ DONNÉES] Rejet bloquant par Great Expectations ({suite_name}) : "
                f"{len(failed_checks)} contrôle(s) non satisfait(s). Rapport archivé à : {fail_file.name}",
                suite_name=suite_name,
                failed_expectations=failed_checks,
                sample_offending_appids=report_summary["sample_offending_appids"],
                report_path=str(fail_file),
            )

    return report_summary
