"""src/tracking/qualification.py - Politique formelle et déterministe de qualification des modèles.

Fournit :
- Moteur d'évaluation pur et déterministe appliquant les règles de qualification.
- Vérification systématique du dépassement des baselines naïves sur la réserve commune de validation.
- Contrôle de finitude des métriques (absence de NaN et d'infinis).
- Règle de provenance : exclusion des modèles synthétiques de la production (statut DEMO_ONLY).
- Production d'un rapport structuré (QualificationReport) archivable et auditable.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

DEFAULT_QUALIFICATION_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "qualification.json"


class RuleCheck(BaseModel):
    """Résultat du contrôle d'une règle unitaire."""

    name: str
    passed: bool
    expected: str
    observed: str
    message: str


class QualificationReport(BaseModel):
    """Rapport officiel d'arbitrage de qualification d'un run candidat."""

    run_id: str
    verdict: Literal["PRODUCTION_READY", "REJECTED", "DEMO_ONLY"]
    passed: bool
    summary: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    checks: List[RuleCheck] = Field(default_factory=list)
    rejection_reasons: List[str] = Field(default_factory=list)
    metrics_snapshot: Dict[str, float] = Field(default_factory=dict)
    tags_snapshot: Dict[str, str] = Field(default_factory=dict)


def load_qualification_policy(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Charge le fichier de configuration de la politique de qualification."""
    target_path = Path(config_path) if config_path else DEFAULT_QUALIFICATION_CONFIG_PATH
    if not target_path.exists():
        raise FileNotFoundError(f"Fichier de configuration de qualification introuvable : {target_path}")
    with open(target_path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_run_qualification(
    run_id: str,
    metrics: Dict[str, float],
    tags: Optional[Dict[str, str]] = None,
    reload_test_passed: bool = True,
    config_path: Optional[str | Path] = None,
) -> QualificationReport:
    """Évalue de façon pure et déterministe si un run MLflow est apte à la promotion."""
    policy = load_qualification_policy(config_path)
    rules_cfg = policy.get("rules", {})
    thresholds_cfg = rules_cfg.get("thresholds", {})
    reg_thresh = thresholds_cfg.get("regression", {})
    cls_thresh = thresholds_cfg.get("classification", {})

    run_tags = tags or {}
    checks: List[RuleCheck] = []
    rejection_reasons: List[str] = []

    # --------------------------------------------------------------------------
    # 1. Vérification des métriques obligatoires
    # --------------------------------------------------------------------------
    mandatory = rules_cfg.get("mandatory_metrics", [])
    missing_metrics = [m for m in mandatory if m not in metrics]
    if missing_metrics:
        checks.append(
            RuleCheck(
                name="mandatory_metrics_present",
                passed=False,
                expected=f"Toutes présentes ({len(mandatory)})",
                observed=f"Manquantes : {missing_metrics}",
                message="Des métriques d'évaluation requises sont absentes.",
            )
        )
        rejection_reasons.append(f"Métriques obligatoires manquantes : {missing_metrics}")
    else:
        checks.append(
            RuleCheck(
                name="mandatory_metrics_present",
                passed=True,
                expected=f"Toutes présentes ({len(mandatory)})",
                observed="Toutes présentes",
                message="Toutes les métriques obligatoires sont renseignées.",
            )
        )

    # --------------------------------------------------------------------------
    # 2. Contrôle de finitude (absence de NaN / Inf)
    # --------------------------------------------------------------------------
    non_finite = []
    for k, v in metrics.items():
        if v is None or math.isnan(v) or math.isinf(v):
            non_finite.append(k)

    if non_finite:
        checks.append(
            RuleCheck(
                name="finite_values_only",
                passed=False,
                expected="Toutes les valeurs réelles et finies",
                observed=f"Valeurs non finies : {non_finite}",
                message="Présence de NaN ou Inf dans les métriques.",
            )
        )
        rejection_reasons.append(f"Métriques non finies détectées : {non_finite}")
    else:
        checks.append(
            RuleCheck(
                name="finite_values_only",
                passed=True,
                expected="Toutes les valeurs réelles et finies",
                observed="100 % valeurs finies",
                message="Toutes les métriques sont numériques et finies.",
            )
        )

    # --------------------------------------------------------------------------
    # 3. Évaluation Régression (MAE vs Baseline, R2)
    # --------------------------------------------------------------------------
    val_reg_mae = metrics.get("regression.val.mae")
    base_reg_mae = metrics.get("regression.val.baseline_mae")
    val_reg_r2 = metrics.get("regression.val.r2")

    if val_reg_mae is not None and base_reg_mae is not None:
        # Règle de surpassement strict de la baseline
        reg_beats_base = val_reg_mae < base_reg_mae
        checks.append(
            RuleCheck(
                name="regression_beats_baseline_mae",
                passed=reg_beats_base,
                expected=f"< baseline ({base_reg_mae:.4f})",
                observed=f"{val_reg_mae:.4f}",
                message="Le candidat bat la baseline naïve de régression." if reg_beats_base else "Échec : le candidat ne surpasse pas la baseline naïve de régression.",
            )
        )
        if not reg_beats_base:
            rejection_reasons.append(
                f"Régression MAE ({val_reg_mae:.4f}) >= Baseline ({base_reg_mae:.4f})"
            )

        # Seuil maximum de MAE
        max_mae = float(reg_thresh.get("max_val_mae", 25.0))
        reg_within_max = val_reg_mae <= max_mae
        checks.append(
            RuleCheck(
                name="regression_max_val_mae",
                passed=reg_within_max,
                expected=f"<= {max_mae}",
                observed=f"{val_reg_mae:.4f}",
                message="MAE dans les limites acceptables." if reg_within_max else f"MAE ({val_reg_mae:.4f}) excède le seuil toléré ({max_mae}).",
            )
        )
        if not reg_within_max:
            rejection_reasons.append(f"Régression MAE ({val_reg_mae:.4f}) > {max_mae}")

    if val_reg_r2 is not None:
        min_r2 = float(reg_thresh.get("min_val_r2", 0.05))
        r2_ok = val_reg_r2 >= min_r2
        checks.append(
            RuleCheck(
                name="regression_min_val_r2",
                passed=r2_ok,
                expected=f">= {min_r2}",
                observed=f"{val_reg_r2:.4f}",
                message="R2 satisfaisant." if r2_ok else f"R2 ({val_reg_r2:.4f}) inférieur au minimum ({min_r2}).",
            )
        )
        if not r2_ok:
            rejection_reasons.append(f"Régression R2 ({val_reg_r2:.4f}) < {min_r2}")

    # --------------------------------------------------------------------------
    # 4. Évaluation Classification (Accuracy et F1-Macro vs Baseline)
    # --------------------------------------------------------------------------
    val_cls_acc = metrics.get("classification.val.accuracy")
    base_cls_acc = metrics.get("classification.val.baseline_accuracy")
    val_cls_f1 = metrics.get("classification.val.f1_macro")
    base_cls_f1 = metrics.get("classification.val.baseline_f1_macro")

    if val_cls_acc is not None and base_cls_acc is not None:
        acc_beats_base = val_cls_acc >= base_cls_acc
        checks.append(
            RuleCheck(
                name="classification_beats_baseline_acc",
                passed=acc_beats_base,
                expected=f">= baseline ({base_cls_acc:.4f})",
                observed=f"{val_cls_acc:.4f}",
                message="Accuracy égale ou supérieure à la baseline." if acc_beats_base else "Accuracy inférieure à la baseline majoritaire.",
            )
        )
        if not acc_beats_base:
            rejection_reasons.append(
                f"Classification Accuracy ({val_cls_acc:.4f}) < Baseline ({base_cls_acc:.4f})"
            )

    if val_cls_f1 is not None and base_cls_f1 is not None:
        f1_beats_base = val_cls_f1 > base_cls_f1
        checks.append(
            RuleCheck(
                name="classification_beats_baseline_f1",
                passed=f1_beats_base,
                expected=f"> baseline ({base_cls_f1:.4f})",
                observed=f"{val_cls_f1:.4f}",
                message="F1-Macro strictement supérieur à la baseline." if f1_beats_base else "F1-Macro inférieur ou égal à la baseline naïve.",
            )
        )
        if not f1_beats_base:
            rejection_reasons.append(
                f"Classification F1-Macro ({val_cls_f1:.4f}) <= Baseline ({base_cls_f1:.4f})"
            )

    # --------------------------------------------------------------------------
    # 5. Contrôle du rechargement externe
    # --------------------------------------------------------------------------
    checks.append(
        RuleCheck(
            name="external_reload_test",
            passed=reload_test_passed,
            expected="True",
            observed=str(reload_test_passed),
            message="Test de rechargement externe validé." if reload_test_passed else "Échec du test de rechargement externe dans un interpréteur vierge.",
        )
    )
    if not reload_test_passed:
        rejection_reasons.append("Échec du test de rechargement externe indépendant.")

    # --------------------------------------------------------------------------
    # 6. Règle de provenance : Détection des flux synthétiques / démo
    # --------------------------------------------------------------------------
    is_synthetic = (
        run_tags.get("pipeline_mode", "").upper() == "SYNTHETIC"
        or run_tags.get("is_synthetic", "").lower() == "true"
        or run_tags.get("synthetic", "").lower() == "true"
        or run_tags.get("demo_only", "").lower() == "true"
    )

    checks.append(
        RuleCheck(
            name="provenance_check",
            passed=True,
            expected="Production-ready ou Démo traçable",
            observed="Synthétique (Démo uniquement)" if is_synthetic else "Historique / Réel authentifié",
            message="Données de démo détectées : promotion bridée à DEMO_ONLY." if is_synthetic else "Données réelles validées.",
        )
    )

    # --------------------------------------------------------------------------
    # 7. Décision Finale
    # --------------------------------------------------------------------------
    all_quality_checks_passed = len(rejection_reasons) == 0

    if not all_quality_checks_passed:
        verdict = "REJECTED"
        passed = False
        summary = f"Candidat rejeté : {len(rejection_reasons)} critère(s) de non-conformité relevé(s)."
    elif is_synthetic:
        verdict = "DEMO_ONLY"
        passed = True
        summary = "Qualité scientifique validée mais données synthétiques : statut DEMO_ONLY réservé."
    else:
        verdict = "PRODUCTION_READY"
        passed = True
        summary = "Toutes les exigences de qualification sont rigoureusement satisfaites : PRODUCTION_READY."

    return QualificationReport(
        run_id=run_id,
        verdict=verdict,
        passed=passed,
        summary=summary,
        checks=checks,
        rejection_reasons=rejection_reasons,
        metrics_snapshot={k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
        tags_snapshot=run_tags,
    )
