# Guide d'Exploitation et d'Administration Système (docs/operations.md)

> **Projet :** Industrialisation de Modèles Steam — MLOps 5MLDE  

Ce document décrit les procédures courantes d'exploitation, de maintenance, de synchronisation et de surveillance de la plateforme Steam MLOps.

---

## 1. Gestion de l'Infrastructure Docker

L'ensemble des services d'infrastructure est conteneurisé et orchestré via `infra/compose.yml`.

### 1.1. Démarrage de l'Infrastructure
```powershell
# Windows PowerShell
cd infra; docker compose up -d --wait; cd ..
```
```sh
# Linux / macOS (Bash)
cd infra && docker compose up -d --wait && cd ..
```

### 1.2. Vérification de l'État de Santé des Conteneurs
```sh
docker compose -f infra/compose.yml ps
```
Les 4 services doivent afficher le statut `healthy` :
- `steam-mlops-infra-postgres-1` (PostgreSQL 18.6 sur port 5432)
- `steam-mlops-infra-mlflow-1` (MLflow 3.16.1 sur port 7281)
- `steam-mlops-infra-prefect-1` (Prefect 3.8.7 sur port 4200)
- `steam-mlops-infra-grafana-1` (Grafana 13.2.3 sur port 3001)

### 1.3. Arrêt Propre Sans Perte de Données
```sh
# Arrêt propre en conservant tous les volumes de persistance
docker compose -f infra/compose.yml stop
```
> [!CAUTION]
> **Interdiction formelle d'exécuter `docker compose down -v` :** L'option `-v` détruit définitivement les volumes Docker contenant la base PostgreSQL `steam_mlops` et le stockage des artefacts MLflow.

---

## 2. Migrations et Schémas Relationnels PostgreSQL

La base de données `steam_mlops` est initialisée via des scripts SQL modulaires idempotents situés dans `infra/postgres/`.

### 2.1. Initialisation Initiale (Schéma `tracing`)
Exécutée automatiquement au premier démarrage de PostgreSQL via le montage Docker `/docker-entrypoint-initdb.d/010-tracing.sql` :
- Déclaration des rôles `api_writer` et `grafana_reader`.
- Création du schéma `tracing` et des tables `model_versions`, `api_requests`, `predictions`, `feedbacks`.

### 2.2. Application des Migrations d'Observabilité (Schéma `ops`)
Pour mettre à jour une base existante avec le schéma de télémétrie des entraînements :
```sh
uv run --locked python scripts/apply_ops_migration.py
```
- Crée le schéma `ops` et les tables `pipeline_runs`, `pipeline_tasks`, `export_sync_state`.
- Crée le rôle à moindre privilège `ops_writer` et accorde les droits `SELECT` à `grafana_reader`.

---

## 3. Synchronisation de la Télémétrie d'Orchestration (`ops_export.py`)

Les traitements Prefect 3 et les publications MLflow stockées en SQLite sont synchronisés vers PostgreSQL par le script `src/orchestration/ops_export.py`.

### 3.1. Exécution Manuelle de l'Exporteur
```sh
uv run --locked python -m src.orchestration.ops_export
```

### 3.2. Propriétés Garanties de l'Exporteur
- **Idempotence stricte :** Aucun doublon lors des réexécutions grâce aux clauses `ON CONFLICT DO UPDATE`.
- **Découpage par lots :** Les requêtes à l'API Prefect sont morcelées par paquets de 10 UUIDs pour respecter les limites du moteur d'orchestration.
- **Audit de fraîcheur :** La table `ops.export_sync_state` enregistre l'horodatage UTC et le nombre de lignes synchronisées à chaque passe.

---

## 4. Administration du Dashboard Grafana

- **URL d'accès :** `http://127.0.0.1:3001`
- **Identifiants par défaut :** `admin` / `admin`
- **Provisioning automatique :** Le tableau de bord `infra/grafana/dashboards/steam-mlops.json` est automatiquement chargé dans le dossier "Operations" sans manipulation manuelle dans l'UI.

### 4.1. Structure des 5 Rangées de Supervision
1. **Traitement & Ingestion :** Volumes de jeux ingérés, états des runs Prefect, débits PUSH/PULL.
2. **Qualité & Barrières :** Taux de succès Great Expectations, motifs de rejet, anomalies détectées.
3. **Inférence & Latence :** Débit global (req/s), latences percentiles $p_{50}, p_{95}, p_{99}$, taux d'erreurs HTTP 4xx/5xx.
4. **Modèles & Gouvernance :** Versions actives en production, identifiants des manifestes, historique des promotions et rollbacks.
5. **Feedbacks & Dérive :** Volumes de retours reçus, ratio d'utilité (`useful`), dérive entre prédictions et vérités terrain réelles.

---

## 5. Surveillance et Sauvegardes

### 5.1. Diagnostic de Santé Global (`scripts/doctor.py`)
À exécuter au début de chaque quart d'exploitation ou après une opération de maintenance :
```sh
uv run --locked python scripts/doctor.py
```

### 5.2. Journal Local de Secours (`state/telemetry_fallback.jsonl`)
En cas de coupure temporaire de PostgreSQL, l'API bascule les traces d'erreurs dans ce fichier JSON Lines append-only.
- **Emplacement :** `state/telemetry_fallback.jsonl`
- **Vérification de taille :**
```powershell
Get-Item state/telemetry_fallback.jsonl | Select-Object Length, LastWriteTime
```
- **Contenu d'une ligne type :**
```json
{"timestamp": "2026-11-16T18:47:01.187Z", "request_id": "b115a324-5c27-4fb5-b76a-f5299cb90158", "endpoint": "/predict", "http_status": 503, "error_code": "DEPENDENCY_UNAVAILABLE", "message": "connection timeout expired"}
```

### 5.3. Sauvegarde Logique de PostgreSQL
```sh
docker compose -f infra/compose.yml exec -T postgres pg_dump -U postgres -d steam_mlops > backups/steam_mlops_$(date +%Y%m%d_%H%M%S).sql
```
