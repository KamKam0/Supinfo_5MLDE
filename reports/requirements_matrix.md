# Matrice de Traçabilité des Exigences

Ce document assure la correspondance biunivoque entre les exigences fonctionnelles et opérationnelles du projet MLOps Steam et les composants techniques développés, ainsi que leurs protocoles de vérification.

---

## 1. Tableau de Traçabilité Globale

| ID | Intitulé Exigence | Module Cible / Emplacement | Protocole de Vérification & Preuve Attendue | Statut |
|---|---|---|---|---|
| **REQ-01** | **Code modulaire** | `src/`, `src/tracking/`, `src/orchestration/`, `api_prediction/`, `api_trigger_passif/` | Revue d'architecture, séparation stricte des 4 fonctions mères, tests unitaires par domaine sans interférence | `DONE` |
| **REQ-02** | **Configuration séparée** | `config/*.json`, `src/settings.py`, `.env.example` | Exécution du projet depuis un répertoire tiers, validation Pydantic des configs, export de la configuration résolue | `IN_PROGRESS` |
| **REQ-03** | **Dépendances maîtrisées** | `pyproject.toml`, `uv.lock`, conteneurs fixés | Installation déterministe via `uv sync --locked`, audit des versions avec freeze complet | `DONE` |
| **REQ-04** | **Tests et installation** | `tests/unit/`, `tests/integration/`, `tests/contracts/`, `README.md` | Exécution complète de `pytest` (100% de tests passants) et protocole d'installation "clone vierge" sans aide externe | `IN_PROGRESS` |
| **REQ-05** | **Absence de dépendance notebook** | `scripts/*.py`, CLI modulaire, APIs | Exécution nominale de bout en bout de l'ingestion, de l'entraînement et de l'inférence sans aucun serveur Jupyter | `DONE` |
| **REQ-06** | **Ingestion reproductible** | `src/ingestion.py`, `src/data_sources/`, `data/snapshots/` | Relecture du snapshot figé, acquisition PULL avec budget réseau, upsert immuable et validation GX | `DONE` |
| **REQ-07** | **Qualité avant prétraitement** | `src/data_validation.py`, Great Expectations | Validation bloquante sur données brutes ; espions prouvant l'absence d'appel de l'entraînement si validation KO | `DONE` |
| **REQ-08** | **Arrêt en non-conformité** | `src/exceptions.py`, pipelines Prefect | Interruption immédiate du pipeline en cas d'anomalie, code retour != 0, modèles existants non écrasés | `DONE` |
| **REQ-09** | **Enchaînement orchestré** | `src/orchestration/tasks.py`, Prefect 3 | Exécution d'un Flow Prefect commun orchestrant les 4 tâches mères, capture du graphe d'états, chaîne de corrélation universelle batch_id -> flow_run_id -> mlflow_run_id -> publication_id | `DONE` |
| **REQ-10** | **Déclenchement automatique** | Triggers Prefect, Seuil PUSH dans `api_trigger_passif/`, `flow.serve` | Seuil d'accumulation PUSH strict à 500 jeux, pipeline orchestré, déploiement planifié 10 min et runner local flow.serve validés | `DONE` |
| **REQ-11** | **Échecs et alertes** | Retries bornés, webhook local et simulateur (`src/orchestration/alerts.py`) | Récepteur HTTP local autonome, transmission d'alertes JSON formatées sur échec sans blocage amont, connecteur Teams sécurisé | `DONE` |
| **REQ-12** | **Suivi complet (Tracking)** | MLflow Tracking Server, `src/tracking/mlflow_tracking.py` | Enregistrement systématique des hyperparamètres, métriques, versions de code, snapshots et artefacts rechargeables | `DONE` |
| **REQ-13** | **Versionnement et règles** | MLflow Model Registry, `src/tracking/registry.py` | Enregistrement Model Registry des modèles, règles d'acceptation automatique et promotion @champion | `DONE` |
| **REQ-14** | **Retour arrière (Rollback)** | Manifeste de publication, script de rollback | Réactivation d'une version antérieure sans réentraînement validée ; scénario de crise complet | `IN_PROGRESS` |
| **REQ-15** | **Prédiction (Inférence)** | `api_prediction/`, `scripts/build_service_manifest.py`, `state/service-manifest.json` | Endpoints `/predict` unitaire et `/predict/batch` pour régression et classification conjointes, chargement au lifespan sans reload par requête, tests validés | `DONE` |
| **REQ-16** | **Validation et erreurs** | `api_prediction/schemas.py`, `src/contracts.py` | Schémas Pydantic stricts (extra='forbid'), rejet formel en HTTP 422 de toute variable cible (pct_pos_total, etc.), codes d'erreurs standardisés | `DONE` |
| **REQ-17** | **Version utilisée** | `api_prediction/main.py`, `infra/postgres/010-tracing.sql` | Présence de `model_version`, `model_name` et `request_id` unique dans chaque réponse HTTP et traçabilité en base PostgreSQL `tracing.model_versions` | `DONE` |
| **REQ-18** | **Feedbacks** | `api_prediction/feedback.py`, `config/label_sources.json` | Réception des retours utilisateurs/vérités terrain sur `POST /feedback`, idempotence stricte, gestion des conflits 409 et cardinalité 1-1 | `DONE` |
| **REQ-19** | **Traces d'usage** | `api_prediction/repository.py`, `infra/postgres/010-tracing.sql` | Enregistrement transactionnel atomique de chaque requête (`api_requests`) et $2N$ prédictions (`predictions`), durées monotones | `DONE` |
| **REQ-20** | **Charge** | `locustfile.py`, scénarios de charge | Campagnes de tirs Locust documentées avec génération de rapports de latence, throughput et erreurs | `TODO` |
| **REQ-21** | **Comparaison et ressources** | Benchmarks 1, 2, 4 workers, profilage CPU/RAM | Table comparative de performance avec saturation CPU/mémoire mesurée sur conteneurs | `TODO` |
| **REQ-22** | **Recommandation chiffrée** | Rapport de dimensionnement technique | Dimensionnement matériel cible étayé par les données expérimentales et hypothèses de charge | `TODO` |
| **REQ-23** | **Dashboard complet** | Tableau de bord Grafana provisionné | Visualisation des volumes d'ingestion, taux de prédiction, distribution des temps de latence, erreurs et retours | `TODO` |
| **REQ-24** | **Démonstration intégrée** | `scripts/demo_full.py`, rapport de soutenance | Scénario de bout en bout : nominal, injection de données invalides, incident simulé, alerte et reprise | `TODO` |
| **REQ-25** | **Dépôt propre** | Racine projet, `.gitignore`, exclusion de secrets/data | Inspection de l'arbre Git prouvant l'absence totale de CSV volumineux, secrets ou caches | `IN_PROGRESS` |
| **REQ-26** | **README complet** | `README.md` exhaustif | Guide de démarrage clair, commandes reproductibles, architecture expliquée, repris à partir d'un clone vierge | `IN_PROGRESS` |
| **REQ-27** | **Synthèse OPS** | `reports/ops_summary.md` | Document de synthèse d'exploitation fondé sur les résultats réels, compromis et limites du système | `TODO` |

---

## 2. Indicateurs Clés de Conformité

- **Nombre total d'exigences :** 27
- **Exigences `DONE` :** 16
- **Exigences `IN_PROGRESS` :** 5
- **Exigences `TODO` :** 6
- **Exigences `BLOCKED` :** 0
