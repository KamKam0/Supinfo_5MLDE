# Synthèse d'Exploitation et d'Observabilité OPS (reports/ops_summary.md)

Ce document constitue le rapport de référence pour la supervision opérationnelle, la télémétrie des traitements et l'audit des indicateurs de la plateforme MLOps Steam.

---

## 1. Architecture d'Observabilité Multi-Couches

L'architecture d'observabilité repose sur une séparation étanche des responsabilités et des privilèges de sécurité au sein de **PostgreSQL 18.6** et de **Grafana 13.2.3** :

```
[ Clients HTTP / Load Balancer ]
               │
               ▼
[ API FastAPI (Port 8001) ] ──────────► [ PostgreSQL 18.6 (Port 5432) ]
  • Role api_writer (INSERT/SELECT)       • Schéma 'tracing' (api_requests, predictions, feedbacks)
  • Horloge monotone (perf_counter)       • Schéma 'ops' (pipeline_runs, pipeline_tasks, sync_state)
               │                                      ▲
[ Exporteur Télémétrie OPS ] ─────────────────────────┤
  • Role ops_writer (INSERT/UPDATE)                   │
  • Synchronisation Prefect/SQLite                    │
                                                      ▼
                                       [ Grafana 13.2.3 (Port 3001) ]
                                         • Role grafana_reader (SELECT lecture seule stricte)
                                         • Dashboard provisionné : 'Steam MLOps — Traitements'
```

### Rôles de Sécurité et Principe de Moindre Privilège
| Rôle Technique | Base / Schémas | Privilèges Accordés | Restrictions Inviolables |
|---|---|---|---|
| `api_writer` | `steam_api` / `tracing` | `CONNECT`, `USAGE`, `SELECT`, `INSERT` | Aucun droit `UPDATE`, `DELETE`, ni accès au schéma `ops` |
| `ops_writer` | `steam_api` / `ops` | `CONNECT`, `USAGE`, `SELECT`, `INSERT`, `UPDATE`, `DELETE` | Aucun accès au schéma `tracing` |
| `grafana_reader` | `steam_api` / `tracing`, `ops` | `CONNECT`, `USAGE`, `SELECT` | `default_transaction_read_only = on`, `statement_timeout = 10s` |

---

## 2. Organisation du Dashboard Grafana (`steam-mlops-overview`)

Le tableau de bord provisionné automatiquement dans le dossier **Operations** est structuré en **5 rangées thématiques** indépendantes :

### Rangée 1 — Pipeline d'Entraînement & Qualité des Données
- **Exécutions de Pipeline :** Volume total d'entraînements orchestrés sous Prefect 3, taux de réussite global et ventilation des statuts terminaux (`COMPLETED`, `FAILED`).
- **Conformité Great Expectations :** Taux de validation conforme des données brutes (`snapshot_integrity_v1`, `push_batch_v1`) et comptage des blocages amont sans retry intempestif.
- **Fraîcheur Télémétrie OPS :** Mesure en temps réel du décalage de synchronisation (`sync_lag_seconds`) entre l'horodatage courant et le dernier export de `ops_export.py`.
- **Historique Détaillé des Flows :** Table d'audit retraçant pour chaque exécution son `flow_run_id`, `batch_id`, mode (`push_threshold`, `manual`), durée et statut GX.

### Rangée 2 — Trafic & Erreurs de l'API d'Inférence
- **Volumétrie HTTP :** Débit global (req/s) et volume cumulé par endpoint (`/predict`, `/predict/batch`, `/feedback`).
- **Jeux Prédits Cumulés :** Somme mathématique de `item_count` sur les requêtes réussies de prédiction (HTTP 200), excluant formellement les requêtes de feedback.
- **Taux d'Erreurs 5xx & 4xx :** Calculé avec rigueur sur le **dénominateur exact du total des tentatives HTTP reçues** :
  $$\text{Taux 5xx (\%)} = 100 \times \frac{\sum \text{requêtes } (\text{status} \ge 500)}{\sum \text{tentatives HTTP reçues}}$$
- **Distribution des Codes Applicatifs :** Répartition des motifs d'erreurs (`INVALID_INPUT`, `DEPENDENCY_UNAVAILABLE`, `PREDICTION_CONFLICT`).

### Rangée 3 — Latences, Percentiles & Activité des Workers
- **Percentiles de Traitement Monotone ($p_{50}, p_{95}, p_{99}$) :** Calculés en SQL natif via `percentile_cont` sur la durée totale `processing_duration_ms` sans jointure cartésienne.
- **Latence d'Inférence Pure Modèles :** Série distincte mesurant le temps d'exécution effectif des estimateurs scikit-learn (`inference_duration_ms`).
- **Activité des Workers :** Nombre de requêtes et latence moyenne ventilés par `instance_id` (UUID unique par worker Uvicorn) et `worker_pid`.
- **Latence par Taille de Lot :** Analyse de la scalabilité vectorielle selon la taille de batch (1 à 100 éléments).

### Rangée 4 — Modèles Servis & Collecte des Feedbacks
- **Inventaire des Modèles en Production :** Volume de prédictions ventilé par nom enregistré (`steam-positive-ratio`, `steam-owner-range`), tâche et numéro de version exact.
- **Volume de Feedbacks Reçus :** Total des retours persistés de manière idempotente sur `POST /feedback`.
- **Satisfaction Utilisateur :** Proportion d'avis favorables (`useful = true`) calculée sur l'ensemble des retours annotés :
  $$\text{Satisfaction (\%)} = 100 \times \frac{\sum (\text{useful} = \text{true})}{\sum (\text{useful IS NOT NULL})}$$

### Rangée 5 — Qualité Observée & Dérive (Vérités Terrain Réelles)
- **Régression Réelle (MAE glissante) :** Écart moyen absolu entre prédictions et observations réelles en points de pourcentage :
  $$\text{MAE}_{\text{observée}} = \frac{1}{K} \sum_{k=1}^K \left| \hat{y}_{\text{pred}} - y_{\text{obs}} \right|$$
  *Règle de rigueur absolue :* Exclusion formelle des sources artificielles (`label_source NOT IN ('loadtest', 'synthetic-course')`).
- **Classification Réelle (Accuracy) :** Taux de concordance exacte des tranches de propriétaires sur retours réels.
- **Couples de Vérité Terrain :** Effectif exact de couples labellisés comparables disponibles.

---

## 3. Résultats de la Campagne d'Audit Étalonnée (`demo-nominal`)

Une séquence étalonnée de 7 tentatives métier a été soumise au système et rapprochée par requêtes SQL directes (`scripts/verify_sql_counts.py`) :

| Métrique Contrôlée | Attendu | Observé SQL | Statut | Règle Validée |
|---|:---:|:---:|:---:|---|
| **Tentatives HTTP totales (`api_requests`)** | **7** | **7** | `OK` | Traçabilité intégrale de chaque tentative métier |
| **Requêtes HTTP 200 (succès prédictions & replay)** | **4** | **4** | `OK` | 2 unitaires + 1 lot vectorisé + 1 replay feedback |
| **Requêtes HTTP 201 (création nouveaux feedbacks)** | **2** | **2** | `OK` | Enregistrement de 2 retours conformes distincts |
| **Requêtes HTTP 422 (rejet fuite de cible)** | **1** | **1** | `OK` | Barrière anti-fuite `extra='forbid'` (injection `pct_pos_total`) |
| **Erreurs HTTP 5xx (indisponibilité)** | **0** | **0** | `OK` | Stabilité nominale du service et du pool SQL |
| **Jeux prédits avec succès (somme `item_count`)** | **5** | **5** | `OK` | 2 unitaires + 3 jeux en lot vectorisé |
| **Lignes de prédiction totales (`predictions`)** | **10** | **10** | `OK` | Règle d'or de cardinalité $2N$ (5 régression + 5 classification) |
| **Prédictions régression (`pct_pos`)** | **5** | **5** | `OK` | Cardinalité $N$ exacte pour la régression |
| **Prédictions classification (`owners`)** | **5** | **5** | `OK` | Cardinalité $N$ exacte pour la classification |
| **Feedbacks persistés uniques (`feedbacks`)** | **2** | **2** | `OK` | Idempotence stricte : le rejeu n'a pas dupliqué l'avis |
| **Violations de cardinalité $2N$** | **0** | **0** | `OK` | Continuité stricte des indices de 0 à $N-1$ |

**Verdict officiel :** 100 % de concordance à l'unité près sans aucun écart.

---

## 4. Limites de Visibilité et Procédures de Diagnostic

Le système d'observabilité documente explicitement ses frontières de mesure pour éviter toute illusion de complétude :

1. **Tentatives non interceptées par l'API :**
   - Une requête bloquée en amont (panne réseau, rejet pare-feu ou saturation des sockets OS avant Uvicorn) n'atteint pas l'application et n'apparaît donc pas dans `tracing.api_requests`.
   - *Remédiation :* Croisement avec les sondes de disponibilité externes et métriques de charge Locust.

2. **Panne d'accès à PostgreSQL :**
   - Si PostgreSQL devient totalement inaccessible, l'API bascule en mode de repli local : les requêtes d'inférence sont rejetées en HTTP 503 (règle d'or d'atomicité) et les événements sont consignés en append-only dans `state/telemetry_fallback.jsonl`.
   - *Diagnostic :* Vérifier le panneau "Fraîcheur Télémétrie OPS" et inspecter `state/telemetry_fallback.jsonl`.

3. **Décalage de synchronisation des traitements :**
   - L'exporteur `src/orchestration/ops_export.py` s'exécute périodiquement. Le panneau de décalage (`sync_lag_seconds`) avertit les opérateurs si le délai dépasse 300 secondes.
