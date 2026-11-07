# Audit Initial des Sources & Décisions d'Architecture

Ce rapport consigne l'origine des sources pédagogiques initiales, les composants réutilisables, les composants refactorisés ou créés *ex nihilo*, ainsi que les décisions d'architecture directrices du projet.

---

## 1. Référence et Origine des Sources Pédagogiques

- **Référentiel source :** Répertoire pédagogique SUPINFO `5MLDE-TPs` (auteur : Baptiste DAVID).
- **Date de figeage :** Novembre 2026.
- **Chemin relatif amont :** `../5MLDE-TPs/`
- **Inventaire des briques pédagogiques amont :**

| Module Amont | Description & Rôle | Utilisation dans le Projet Final |
|---|---|---|
| `TP00_setup_des_postes` | `compose.yml`, `.env.example`, guide d'installation | Socle des services Docker (MLflow, Prefect, PostgreSQL, Grafana). Fichiers adaptés pour isoler les ports et les volumes. |
| `TP01_industrialisation` | `pyproject.toml`, `uv.lock`, fonctions de découpage et modèles initiaux (`src/data.py`, `src/models.py`) | Base pour la modularisation stricte en 4 fonctions mères, formalisation des contrats Pydantic et élimination définitive du notebook. |
| `TP02_donnees_dynamiques` | `simulateur_push.py`, squelette `api_trigger_passif/` | Conception de l'API de collecte PUSH isolée, buffer persistant et trigger automatique à seuil. |
| `TP03_mlflow` | Protocoles de tracking et registry MLflow | Implémentation du tracking complet, signatures de modèles, règles d'acceptation et politique de rollback. |
| `TP04_prefect` | Orchestration Prefect, flux de tâches et alertes | Architecture du Flow Prefect unifié, retries bornés et gestion des alertes d'incidents. |
| `TP05_api_supervision` | Schéma SQL `init-db.sql`, dashboard Grafana, tests Locust | Implémentation de l'API d'inférence FastAPI, journalisation PostgreSQL transactionnelle, métriques de latence et dashboard opérationnel. |

---

## 2. Décisions Fondatrices d'Architecture

### Décision 1 — Exécution Intégralement Locale
Le projet est conçu pour s'exécuter localement sur un poste de développement sous Windows (ou conteneurs Linux WSL 2).
- Aucun service cloud externe payant n'est requis (AWS, GCP, Azure).
- Aucun cluster Kubernetes n'est imposé pour la validation du socle.
- Tous les services (PostgreSQL 16+, MLflow 2.x, Prefect 3.x, Grafana) tournent via Docker Compose local ou en processus Python natifs gérés par `uv`.

### Décision 2 — Séparation Stricte des Deux APIs (Collecte vs Inférence)
Afin d'éviter tout mélange des flux opérationnels :
- **API Collecte / Buffer PUSH (port 8001) :** Reçoit des observations labellisées destinées aux réentraînements ou vérifications de dérive. Elle écrit dans un stockage durable et déclenche un traitement Prefect au franchissement d'un seuil.
- **API Inférence Prédictive (port 8000) :** Reçoit uniquement des caractéristiques de jeux (sans variable cible), valide le schéma d'entrée, appelle le modèle champion en cache mémoire, journalise la requête et la prédiction dans PostgreSQL, et gère les feedbacks asynchrones.

### Décision 3 — Cloisonnement et Non-Fuite des Données (Data Splitting)
- Découpage figé au niveau des identifiants `appid` :
  - **Entraînement / Apprentissage :** Jeux autorisés aux entraînements successifs.
  - **Qualification (15%) :** Ensemble figé servant à l'évaluation impartiale des candidats.
  - **Test Final (15%) :** Ensemble hermétique strictement réservé à la validation finale sans influence sur le choix du modèle.
- **Règle absolue :** Aucun `appid` appartenant à la qualification ou au test ne doit jamais entrer dans un lot d'entraînement.

### Décision 4 — Gestion des Dépendances et Reproductibilité avec `uv`
- Utilisation exclusive de l'outil `uv` avec verrouillage déterministe (`uv.lock`).
- Version Python cible : `Python 3.12` ou `Python 3.13` (environnement Windows PowerShell natif pour l'interpréteur, WSL 2 pour le moteur Docker).
- Interdiction d'ajouter des dépendances sans mise à jour immédiate du lockfile (`uv sync --locked`).

---

## 3. Inventaire Technique des Accès et Prérequis

| Composant / Dépendance | État Observé | Impact / Action |
|---|---|---|
| **Python** | `Python 3.13.14` actif | Compatible. Configuration de l'environnement virtuel avec `uv`. |
| **Gestionnaire `uv`** | `uv 0.13.6` présent | Opérationnel pour la résolution des dépendances et scripts CLI. |
| **Git** | `git version 2.48.1` | Dépôt local initialisé (`main`). Hygiène `.gitignore` validée. |
| **Docker Engine & Compose** | Docker v29.9.2 actif | Moteur de conteneurs opérationnel. |
| **Ports réseau :** | | |
| - Port 8000 (API Inférence) | Libre | Disponible. |
| - Port 8001 (API Collecte) | Libre | Disponible. |
| - Port 4200 (Prefect UI) | Libre | Disponible. |
| - Port 5000 / 7281 (MLflow UI) | Libre | Disponible. |
| - Port 5432 (PostgreSQL) | Libre sur l'hôte | Disponible pour le bind Compose. |
| - Port 3000 (Grafana) | Port occupé par un conteneur tiers (`nginx-dev`) | **Attention :** Prévoir un remappage (ex : port 3001 sur l'hôte ou arrêt du conteneur tiers) si le conteneur dev reste actif lors du déploiement de Grafana. |
| **Source Dataset Steam** | Présente localement | Le fichier brut `games_march2025_full.csv` (471 Mo) est présent dans `..\TP1\data\`. Pas de blocage lié au téléchargement Kaggle. |
| **Alerting Webhook** | Simulable localement | Webhook Teams optionnel ; mise en place prioritaire d'un récepteur local / logging structuré pour garantir l'autonomie totale sans compte tiers. |
