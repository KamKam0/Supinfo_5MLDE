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
| - Port 8000 (API Collecte PUSH) | Libre | Disponible. |
| - Port 8001 (API Inférence) | Libre | Disponible. |
| - Port 4200 (Prefect UI) | Port 4200 actif (healthy) | Validé. |
| - Port 7281 (MLflow UI) | Port 7281 actif (healthy) | Validé. |
| - Port 5432 (PostgreSQL) | Port 5432 actif (healthy) | Validé (schéma tracing initialisé). |
| - Port 3001 (Grafana) | Port 3001 actif (healthy) | Mappé sur 3001 suite à conflit port 3000 tiers. |
| **Source Dataset Steam** | Présente et validée | `data/raw/games_march2025_full.csv` (471 492 150 octets). SHA-256 : `1ff3b4fa2eeeb7faf1d9c37abb63b9522813532fbf68f4c4a236d5a466688db7` conforme au bit près. |
| **Alerting Webhook** | Simulable localement | Webhook Teams optionnel ; mise en place prioritaire d'un récepteur local / logging structuré pour garantir l'autonomie totale sans compte tiers. |

---

## 4. Audit Approfondi du Dataset Historique & Qualité des Données

### 4.1. Métriques Globales et Empreintes Cryptographiques
- **Source brute :** `games_march2025_full.csv` (471 492 150 octets).
  - Empreinte SHA-256 constatée : `1ff3b4fa2eeeb7faf1d9c37abb63b9522813532fbf68f4c4a236d5a466688db7` (100% conforme à la référence documentaire).
- **Projection canonique (16 colonnes) :** `data/snapshots/historical_projection.csv` (11 611 348 octets, 94 948 lignes).
  - Empreinte SHA-256 constatée : `7eec2ff4e6f1389bfb74308c8de1206fee2a94f51120895c2cf7fce84146312f` (100% conforme à la référence pédagogique).
- **Unicité des identifiants :** 94 948 `appid` strictement uniques (`is_unique = True`).

### 4.2. Analyse des Valeurs Manquantes et Sentinelles
- **Noms de jeux (`name`) :** Exactement 2 valeurs nulles constatées. N'impacte pas l'apprentissage (la feature textuelle `name` est exclue des prédicteurs).
- **Prix (`price`) :** 0 valeur manquante. 19 420 jeux gratuits (`price == 0.0`).
- **Âge requis (`required_age`) :** 1 seule ligne avec la sentinelle `-1` (appid 325140), traitée comme valeur manquante.
- **Succès (`achievements`) et DLC (`dlc_count`) :** 0 valeur négative.
- **Sentinelle avis (`pct_pos_total = -1`) :** 39 575 jeux sans pourcentage d'avis positif disponible.
  - Correspond rigoureusement aux jeux ayant strictement moins de 10 avis (`num_reviews_total < 10`).
- **Sentinelle propriétaires (`estimated_owners = '0 - 0'`) :** 13 656 jeux conventionnellement exclus pour la tâche de classification faute de signal commercial.

### 4.3. Éligibilité par Tâche Prédictive
1. **Régression (`pct_pos_total`) :**
   - Règle : `pct_pos_total != -1` ET `num_reviews_total >= 10`.
   - Effectif éligible : **55 373 jeux** (soit 58,3 % du catalogue brut).
2. **Classification (`estimated_owners`) :**
   - Règle : `estimated_owners != '0 - 0'` ET effectif de classe $\ge 5$.
   - Effectif éligible : **81 290 jeux** répartis sur **12 classes**.
   - Deux classes ultra-minoritaires exclues (1 jeu chacune) : `100000000 - 200000000` et `200000000 - 500000000`.

### 4.4. Limites Scientifiques & Biais Documentés
- **Nature des estimations :** Les tranches de propriétaires reposent sur la méthodologie d'échantillonnage de SteamSpy (profils publics), et non sur les ventes directes certifiées par Valve.
- **Biais de survie et rétroactivité :** Les métadonnées observées constituent une photographie statique (mars 2025). Elles ne permettent pas d'évaluer la prédictibilité *ex-ante* avant commercialisation.
- **Incertitude de devise :** La devise des prix historiques n'est pas explicitement spécifiée dans l'en-tête (considérée conventionnellement comme USD).
