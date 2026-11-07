# Plateforme Industrielle MLOps — Catalogue Steam (Projet 5MLDE)

[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Package Manager](https://img.shields.io/badge/uv-0.13+-brightgreen.svg)](https://astral.sh/uv)
[![Orchestration](https://img.shields.io/badge/Prefect-3.8+-informational.svg)](https://www.prefect.io/)
[![Tracking](https://img.shields.io/badge/MLflow-2.x-orange.svg)](https://mlflow.org/)
[![API](https://img.shields.io/badge/FastAPI-0.115+-teal.svg)](https://fastapi.tiangolo.com/)
[![Database](https://img.shields.io/badge/PostgreSQL-16+-blue.svg)](https://www.postgresql.org/)
[![Monitoring](https://img.shields.io/badge/Grafana-11+-orange.svg)](https://grafana.com/)

Ce projet constitue le socle complet d'industrialisation MLOps pour l'analyse, l'entraînement, la qualification, le déploiement et la supervision de modèles de Machine Learning appliqués au catalogue de jeux de la plateforme **Steam**.

---

## 1. Périmètre & Problématiques Métier

Le système exploite le dataset exhaustif des jeux Steam (`games_march2025_full.csv`) pour adresser **deux tâches prédictives complémentaires** :

1. **Tâche de Régression Continue — Pourcentage d'avis positifs (`pct_pos_total`) :**
   - **Cible :** Valeur numérique réelle comprise dans l'intervalle $[0.0, 100.0]$.
   - **Critère d'éligibilité :** Jeux disposant d'une représentativité statistique minimale (`num_reviews_total >= 10`).
   - **Objectif :** Estimer la satisfaction globale de la communauté à partir des métadonnées du titre (prix, genres, langues, tags, caractéristiques techniques).
   - **Baseline de référence :** Prédicteur naïf médian.
   - **Modèle de référence :** `RandomForestRegressor`.

2. **Tâche de Classification Multiclasse — Tranche de propriétaires estimés (`estimated_owners`) :**
   - **Cible :** Catégorie ordonnée discrète (ex. `20000 - 50000`, `50000 - 100000`, `100000 - 200000`, etc.).
   - **Critère d'éligibilité :** Exclusion standard de la tranche sans signal `0 - 0` et sélection des classes comptant au moins 5 jeux dans le corpus d'ajustement.
   - **Objectif :** Classifier l'envergure commerciale d'un jeu vidéo sans avoir accès aux chiffres de vente directs.
   - **Baseline de référence :** Prédicteur naïf majoritaire (*dummy classifier*).
   - **Modèle de référence :** `RandomForestClassifier`.

---

## 2. Architecture Globale du Système

Le système est conçu selon les principes de l'ingénierie logicielle robuste et des pratiques MLOps avancées :

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 CYCLE MLOPS STEAM                                      │
└────────────────────────────────────────────────────────────────────────────────────────┘

    [ Source Brute CSV ] ──► [ Projection 16 Colonnes ] ──► [ Snapshot SHA-256 Immuable ]
                                                                       │
    ┌──────────────────────────────────────────────────────────────────┘
    ▼
[ Barrière Qualité Great Expectations ] ──(Échec)──► [ Blocage Immédiat + Alerte ]
    │ (Succès)
    ▼
[ Partitions sans Fuite ] ──► Train (70%) / Qualification Figée (15%) / Test Hermétique (15%)
    │
    ▼
[ Feature Engineering ] ──► Extraction Textuelle, One-Hot / Target Encoding, Scalers
    │
    ▼
[ Entraînement & Baselines ] ──► MLflow Tracking (Paramètres, Métriques, Artefacts)
    │
    ▼
[ Évaluation & Décision ] ──► Promotion `@champion` si Delta vs Baseline validé
    │
    ▼
[ Manifeste de Service ] ──► Déploiement Inférence FastAPI (Port 8001)
    ▲
    │
┌───┴────────────────────────────────────────────────────────────────────────────────────┐
│                               SERVICES D'EXPLOITATION                                  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ • API Collecte PUSH (Port 8000)      : Ingestion durable & buffer seuil (500 jeux)     │
│ • API Inférence (Port 8001)          : Prédictions unitaire & batch (sans variables Y) │
│ • Base PostgreSQL (Port 5432)        : Audit transactionnel des requêtes & feedbacks   │
│ • Orchestration Prefect (Port 4200)  : Flow unifié des 4 fonctions mères + retries    │
│ • Tracking MLflow (Port 7281)        : Registre d'expériences et modèles versionnés    │
│ • Supervision Grafana (Port 3000)    : Métriques de latence, volumes, erreurs, dérive │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Prise en Main Rapide (Quickstart)

### 4.1. Prérequis Système
- **Système d'exploitation :** Windows 10/11 (avec WSL 2 pour Docker) ou Linux/macOS.
- **Python :** Version 3.12 ou 3.13 installée.
- **Gestionnaire `uv` :** Indispensable pour la reproductibilité (`uv --version`).
- **Docker & Docker Compose :** Actif pour les services de stockage et supervision.

### 4.2. Installation de l'Environnement
```powershell
# Cloner le dépôt et se placer à la racine
cd "C:\Users\kambj\Desktop\Supinfo\5MLDE\Final Project"

# Synchroniser les dépendances Python via le lockfile déterministe
uv sync --locked

# Vérifier la conformité de l'environnement matériel et logiciel
uv run scripts/doctor.py
```

### 4.3. Démarrage de la Stack d'Infrastructure
```powershell
# Démarrer les services (PostgreSQL, MLflow, Prefect, Grafana)
docker compose up -d --wait

# Vérifier la bonne santé des conteneurs
docker compose ps
```

---

## 4. Traçabilité des Exigences

- **Matrice des 27 exigences :** Consulter [`reports/requirements_matrix.md`](file:///C:/Users/kambj/Desktop/Supinfo/5MLDE/Final%20Project/reports/requirements_matrix.md).
- **Audit des sources & architecture :** Consulter [`reports/source_audit.md`](file:///C:/Users/kambj/Desktop/Supinfo/5MLDE/Final%20Project/reports/source_audit.md).
