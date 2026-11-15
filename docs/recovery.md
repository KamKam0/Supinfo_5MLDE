# Runbook d'Exploitation et Procédures de Reprise après Incident (docs/recovery.md)

> **Projet :** Industrialisation de Modèles Steam — MLOps 5MLDE  

Ce document fournit aux équipes d'exploitation (SRE, MLOps, DevOps) les procédures pas à pas, déterministes et vérifiables pour diagnostiquer, contenir et rétablir le service en cas de panne ou d'anomalie critique sur la plateforme Steam MLOps.

---

## 1. Matrice Synoptique des Incidents et Actions Réflexes

| Identifiant | Scénario d'Incident | Criticité | Détection & Alerte | Impact Service | Action Réflexe Immédiate |
|---|---|---|---|---|---|
| **INC-01** | Données corrompues / Dérive schéma | P2 (Majeur) | Alerte GX, Task `task_load_data` FAILED | Blocage de l'entraînement, service inchangé | Isoler le snapshot dans `data/corrupted/`, inspecter le rapport GX |
| **INC-02** | Indisponibilité PostgreSQL | P1 (Critique) | Sonde `/health/ready` 503, log local | Rejet HTTP 503 sur `/predict` et `/feedback` | Vérifier `docker compose -f infra/compose.yml ps postgres`, relancer sans effacer les volumes |
| **INC-03** | Publication MLflow partielle | P2 (Majeur) | Journal SQLite `status='IN_PROGRESS'` | Modèles non déployés, version asymétrique | Exécuter `scripts/reconcile.py --publication-id <id>` |
| **INC-04** | Dégradation de modèle en production | P1 (Critique) | Alertes métriques, hausses de retours négatifs | Inférences de mauvaise qualité servies | Déclencher le rollback immédiat via `service-manifest.prev.json` |
| **INC-05** | Panne du récepteur d'alertes | P3 (Mineur) | Log `CONNECTION_FAILED` dans le dispatcher | Perte temporaire de notifications | Confinement automatique, vérifier le webhook destinataire |

---

## 2. Procédures Détaillées par Type d'Incident

### 2.1. Runbook INC-01 — Rejet Bloquant de Données (Great Expectations)

#### 1. Symptômes & Détection
- Le pipeline d'entraînement Prefect s'interrompt avec l'erreur `DataValidationError`.
- La tâche mère `task_load_data` passe à l'état `FAILED`.
- Aucune exécution des étapes de prétraitement (`task_preprocess_data`), d'entraînement ou de sélection.

#### 2. Diagnostic Rapide
Consulter le dernier rapport d'anomalie Great Expectations généré dans `reports/validation/` :
```powershell
# PowerShell
Get-ChildItem -Path reports/validation -Filter "validation_failure_*.json" | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | Get-Content | ConvertFrom-Json
```
```sh
# Bash
cat $(ls -t reports/validation/validation_failure_*.json | head -1) | jq .
```

#### 3. Procédure de Rétablissement
1. **Conserver le fichier incriminé** : Ne jamais écraser ni supprimer les données brutes corrompues (elles constituent la pièce justificative d'audit).
2. **Identifier la violation** :
   - Type non conforme (ex. string dans la colonne `price`).
   - Valeur hors bornes (ex. `pct_pos_total` > 100 ou < 0).
   - Duplicata strict sur la clé primaire métier (`appid`).
   - Tranche d'utilisateurs (`estimated_owners`) mal formatée ou inversée.
3. **Corriger la source de données** ou le parseur de scraping amont (`src/data_sources/normalization.py`).
4. **Relancer le pipeline d'ingestion** sur un nouveau batch identifié :
   ```sh
   uv run --locked python -m src.orchestration.flow --batch-id batch-correction-01
   ```
5. **Rapport d'audit de référence :** Consulter [incident_invalid_data.md](file:///C:/Users/kambj/Desktop/Supinfo/5MLDE/Final%20Project/reports/incidents/incident_invalid_data.md).

---

### 2.2. Runbook INC-02 — Crash ou Indisponibilité de PostgreSQL

#### 1. Symptômes & Détection
- La sonde `/health/live` répond HTTP 200 (le processus API est vivant).
- La sonde `/health/ready` répond **HTTP 503 Service Unavailable** (`"Base de données PostgreSQL inaccessible"`).
- Les appels `/predict` et `/feedback` échouent en HTTP 503 avec l'erreur `DEPENDENCY_UNAVAILABLE`.
- **Règle absolue :** Aucun faux code 200 n'est renvoyé aux clients. Les requêtes refusées sont consignées localement dans `state/telemetry_fallback.jsonl`.

#### 2. Diagnostic Rapide
Vérifier l'état du conteneur PostgreSQL :
```sh
docker compose -f infra/compose.yml ps postgres
docker compose -f infra/compose.yml logs --tail 50 postgres
```

#### 3. Procédure de Rétablissement
1. **RÈGLE CRITIQUE D'INTÉGRITÉ :** **Ne JAMAIS exécuter `docker compose down -v`** (destruction irréversible du volume de persistance des prédictions).
2. Redémarrer le conteneur :
   ```sh
   docker compose -f infra/compose.yml start postgres
   ```
3. Vérifier la réouverture du port et la disponibilité du moteur :
   ```sh
   docker compose -f infra/compose.yml exec postgres pg_isready -U postgres -d steam_mlops
   ```
4. **Vérification de la reconnexion automatique :**
   Le pool applicatif `DatabasePool` se reconnecte sans aucun redémarrage d'Uvicorn :
   ```sh
   curl -s http://127.0.0.1:8001/health/ready
   # Doit retourner HTTP 200 : {"status": "ready", ...}
   ```
5. **Rejeu de la télémétrie locale :**
   Inspecter `state/telemetry_fallback.jsonl` pour réconcilier les identifiants de requêtes tombées pendant la panne.
6. **Rapport d'audit de référence :** Consulter [incident_postgres_down.md](file:///C:/Users/kambj/Desktop/Supinfo/5MLDE/Final%20Project/reports/incidents/incident_postgres_down.md).

---

### 2.3. Runbook INC-03 — Publication Interrompue dans MLflow (Asymétrie Régression / Classification)

#### 1. Symptômes & Détection
- Une publication de modèles s'est interrompue à la suite d'un arrêt conteneur, d'une coupure réseau ou d'un timeout pendant l'enregistrement dans le Model Registry.
- Dans `state/publications.db`, la publication est à l'état `status = 'IN_PROGRESS'`.
- Une seule des deux tâches (régression ou classification) a été enregistrée ou a reçu son alias.

#### 2. Diagnostic Rapide
Inspecter les publications interrompues dans la base SQLite locale :
```sh
uv run --locked python scripts/reconcile.py --scan-pending
```

#### 3. Procédure de Réconciliation Idempotente
1. Exécuter le moteur de réconciliation sur l'identifiant de publication concerné :
   ```sh
   uv run --locked python scripts/reconcile.py --publication-id <PUB_ID>
   ```
2. **Propriétés garanties du moteur :**
   - Il recherche les versions déjà créées pour le run MLflow associé (`find_existing_model_version_for_run`).
   - Il **ne recrée aucun doublon** pour le modèle déjà enregistré.
   - Il complète exclusivement l'étape manquante (enregistrement du second modèle et affectation conjointe de l'alias).
   - Il met à jour le statut final à `COMPLETED`.
3. Vérifier les alias dans le Model Registry MLflow (Port 7281) :
   - `http://127.0.0.1:7281/#/models/steam-positive-ratio`
   - `http://127.0.0.1:7281/#/models/steam-owner-range`
4. **Rapport d'audit de référence :** Consulter [incident_mlflow_reconcile.md](file:///C:/Users/kambj/Desktop/Supinfo/5MLDE/Final%20Project/reports/incidents/incident_mlflow_reconcile.md).

---

### 2.4. Runbook INC-04 — Dégradation de Performance et Procédure de Rollback

#### 1. Symptômes & Détection
- Dérive statistique observée sur les résidus de prédiction.
- Signalements anormaux d'utilisateurs (`useful = False` répétés sur l'API `/feedback`).
- Métriques dégradées sur un nouveau modèle récemment promu au statut `@champion`.

#### 2. Procédure d'Urgence de Retour Arrière (Rollback)

La procédure de retour arrière s'effectue **sans aucun réentraînement** en réactivant le couple de modèles immédiatement antérieur.

##### Option A : Rollback Automatisé via la Sauvegarde de Manifeste
```powershell
# 1. Rétablir l'ancien manifeste de service validé
Copy-Item state/service-manifest.prev.json state/service-manifest.json -Force

# 2. Réaligner l'alias @champion dans MLflow Registry
uv run --locked python scripts/select_models.py --rollback 12 --reason "Rollback d'urgence suite dégradation constatée"

# 3. Redémarrer les workers Uvicorn de l'API de prédiction
# (Si lancé via service systemd, docker ou processus local, envoyer signal HUP ou redémarrer)
```

##### Option B : Rollback via la CLI de Sélection
```sh
uv run --locked python scripts/select_models.py --rollback <VERSION_CIBLE> --reason "Anomalie constatée sur version N"
uv run --locked python scripts/build_service_manifest.py --alias champion
```

#### 3. Contrôle & Audit Post-Rollback
1. Interroger `/models` sur l'API pour confirmer que le `manifest_id` et les versions actives correspondent à la version restaurée :
   ```sh
   curl -s http://127.0.0.1:8001/models | jq .
   ```
2. Émettre une prédiction de vérification :
   ```sh
   curl -s -X POST http://127.0.0.1:8001/predict -H "Content-Type: application/json" -d '{"game": {"appid": 1086940, "name": "Baldur'\''s Gate 3", "price": 59.99, "required_age": 18, "genres": ["RPG"], "windows": true, "mac": true, "linux": false, "release_date": "2023-08-03"}}' | jq .
   ```
3. **Vérification de l'intégrité dans PostgreSQL :**
   Les requêtes passées sous le modèle dégradé **ne sont pas supprimées** ; les nouvelles requêtes tracent à nouveau le modèle sain.
   ```sql
   SELECT m.model_name, m.model_version, count(p.prediction_id)
   FROM tracing.predictions p
   JOIN tracing.model_versions m ON p.model_id = m.model_id
   GROUP BY m.model_name, m.model_version
   ORDER BY count(p.prediction_id) DESC;
   ```
4. **Rapport d'audit de référence :** Consulter [incident_model_rollback.md](file:///C:/Users/kambj/Desktop/Supinfo/5MLDE/Final%20Project/reports/incidents/incident_model_rollback.md).

---

### 2.5. Runbook INC-05 — Panne du Système d'Alertes

#### 1. Symptômes & Détection
- Les logs de Prefect ou de l'application affichent un avertissement :
  `[ALERT] Impossible de joindre le récepteur local (http://127.0.0.1:9000/alerts) : CONNECTION_FAILED`.
- Le canal Microsoft Teams ne reçoit pas les cartes adaptatives d'erreur.

#### 2. Diagnostic & Confinement
1. Le dispatcheur (`AlertDispatcher`) capture l'erreur et applique un timeout strict (1 seconde) :
   - Le pipeline d'entraînement **n'est pas bloqué**.
   - **Aucune boucle de relance infinie** n'est déclenchée par un échec de webhook.
2. Vérifier si le récepteur autonome d'alerte local est démarré sur le port 9000 :
   ```powershell
   Test-NetConnection -ComputerName 127.0.0.1 -Port 9000
   ```
3. Si Teams est utilisé, vérifier la variable d'environnement `TEAMS_WEBHOOK_URL` et le flag `TEAMS_ALERT_ENABLED=true`.

#### 3. Rétablissement
1. Démarrer ou relancer le service d'écoute des alertes :
   ```sh
   uv run --locked python -m src.orchestration.alerts
   ```
2. Tester unitairement l'émission d'une alerte sans lancer d'entraînement :
   ```sh
   uv run --locked python scripts/demo_incident.py --scenario alert-down
   ```
3. **Rapport d'audit de référence :** Consulter [incident_alert_failure.md](file:///C:/Users/kambj/Desktop/Supinfo/5MLDE/Final%20Project/reports/incidents/incident_alert_failure.md).

---

## 3. Outils et Commandes de Diagnostic Global

### 3.1. Diagnostic de Santé Global (`scripts/doctor.py`)
Avant toute intervention, exécuter le diagnostic d'infrastructure :
```sh
uv run --locked python scripts/doctor.py
```
Vérifie la présence et la santé des 4 briques :
- PostgreSQL (Port 5432)
- MLflow Tracking Server (Port 7281)
- Prefect Orchestrator (Port 4200)
- Grafana Supervision (Port 3001)

### 3.2. Exécution de la Suite Complète de Démonstration d'Incidents
Pour reproduire et attester l'ensemble des 5 scénarios d'un bloc avec génération automatique des rapports d'audit :
```sh
uv run --locked python scripts/demo_incident.py --scenario all
```
Tous les rapports Markdown correspondants sont immédiatement archivés et consultables dans `reports/incidents/`.
