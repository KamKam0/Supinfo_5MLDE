"""src/exceptions.py - Exceptions personnalisées du projet MLOps Steam.

Définit la hiérarchie d'exceptions levées en cas de violation des règles
de qualité de données (Great Expectations) ou d'étanchéité des flux.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


class SteamMLOpsError(Exception):
    """Exception racine pour l'ensemble du projet MLOps."""
    pass


class DataValidationError(SteamMLOpsError):
    """Levée lorsqu'une barrière de validation de qualité des données échoue.

    Bloque immédiatement le pipeline d'ingestion/entraînement, empêche toute mise à jour
    de modèle en production et archive un rapport d'anomalie.
    """

    def __init__(
        self,
        message: str,
        suite_name: Optional[str] = None,
        failed_expectations: Optional[List[Dict[str, Any]]] = None,
        sample_offending_appids: Optional[List[Any]] = None,
        report_path: Optional[Path | str] = None,
    ) -> None:
        super().__init__(message)
        self.suite_name = suite_name or "unknown_suite"
        self.failed_expectations = failed_expectations or []
        self.sample_offending_appids = sample_offending_appids or []
        self.report_path = str(report_path) if report_path else None

    def to_dict(self) -> Dict[str, Any]:
        """Convertit l'erreur et son diagnostic en dictionnaire exploitable."""
        return {
            "error_type": self.__class__.__name__,
            "message": str(self),
            "suite_name": self.suite_name,
            "failed_expectations_count": len(self.failed_expectations),
            "failed_expectations": self.failed_expectations,
            "sample_offending_appids": self.sample_offending_appids,
            "report_path": self.report_path,
        }


class DataIntegrityError(DataValidationError):
    """Levée lors d'une altération cryptographique ou d'une incohérence structurelle du snapshot."""
    pass


class DataLeakageError(DataValidationError):
    """Levée lors de la détection d'appids réservés dans le jeu d'entraînement ou de cibles à l'inférence."""
    pass


class DataSchemaError(DataValidationError):
    """Levée lorsque les colonnes obligatoires ou leurs types fondamentaux sont non conformes."""
    pass
