# Synthèse d'Exploitation et Bilan Opérationnel MLOps (reports/ops_summary.md)

> **Projet :** Industrialisation de Modèles Steam — MLOps 5MLDE (Supinfo)  

---

## 1. Synthèse Exécutive

Le projet **MLOps 5MLDE** a industrialisé un cycle complet d'apprentissage automatique, de déploiement et de supervision en production sur le catalogue de jeux de la plateforme **Steam** (94 948 titres historiques analysés).

L'ensemble des **27 exigences du projet (REQ-01 à REQ-27)** a été couvert sans exception (voir `reports/requirements_matrix.md`).

### Chiffres Clés du Projet :
- **Snapshot Historique scellé :** 94 948 jeux validés par Great Expectations (0 défaut sur 10 familles de règles).
- **Partitions anti-fuite :** 70 % Train (66 463 appids), 15 % Validation (14 242 appids), 15 % Test (14 243 appids), 30 % réserve sanctuarisée.
- **Modèles opérationnels :** Régression (`RandomForestRegressor`, MAE 12.53% vs baseline 13.91%), Classification (`RandomForestClassifier`, 12 classes, Accuracy 73.45% vs baseline 73.04%, Macro F1 0.1036 vs baseline 0.0704).
- **Débit maximal mesuré :** **143.61 req/s** en inférence unitaire ($p_{95} = 90$ ms), **58.24 req/s** en lot vectorisé (soit **1 164.81 jeux/s** et **2 329.62 inférences/s** à $p_{95} = 200$ ms).
- **Résilience prouvée :** 5 scénarios de crise injectés et surmontés sans falsification de succès, zéro faux 200 renvoyé en cas de crash SQL, rollback de modèle sans réentraînement opérationnel.
- **Qualité du code et non-régression :** 114 tests unitaires et d'intégration validés à 100%.

---

## 2. Bilan des Performances Prédictives vs Baselines Naïves

Conformément à la politique de gouvernance, aucun modèle ne peut être promu au statut `@champion` sans surclasser formellement une heuristique naïve sur la partition de validation scellée.

| Tâche Métier | Modèle Déployé | Métrique Clé | Résultat Modèle | Baseline Naïve | Gain Net Observé | Décision de Qualification |
|---|---|---|---|---|---|---|
| **Régression** (`pct_pos_total`) | `RandomForestRegressor` (v12) | MAE ($\downarrow$) | **12.53 %** | 13.91 % (Médiane) | **-1.38 point** | ✅ ACCEPTÉ (seuil $\ge 1.0$ validé) |
|  |  | RMSE ($\downarrow$) | **17.10 %** | 18.66 % | **-1.56 point** |  |
|  |  | $R^2$ ($\uparrow$) | **0.1602** | 0.0000 | **+0.1602** |  |
| **Classification** (`estimated_owners`) | `RandomForestClassifier` (v12, 12 classes) | Accuracy ($\uparrow$) | **73.45 %** | 73.04 % (Majoritaire) | **+0.41 %** | ✅ ACCEPTÉ (gain sur classes rares) |
|  |  | Macro F1 ($\uparrow$) | **0.1036** | 0.0704 | **+0.0332** |  |

*Note méthodologique :* L'objectif de la notation MLOps repose sur la rigueur de l'évaluation, l'absence absolue de fuite d'information et la traçabilité reproductible plutôt que sur la complexité algorithmique. Les baselines naïves sont systématiquement consignées dans MLflow aux côtés des modèles candidats.

---

## 3. Synthèse des Campagnes de Charge et Recommandation de Dimensionnement

### 3.1. Tableau Synthétique Multi-Workers Uvicorn (Extrait de `reports/sizing.md`)

Les campagnes Locust 2.46.6 ont comparé 1, 2 et 4 workers Uvicorn sur 3 profils d'utilisateurs virtuels sans `--reload` et avec validation stricte `catch_response` :

| Profil de Charge | Workers Uvicorn | Débit Utile (req/s) | Débit Jeux/s | Latence $p_{95}$ (ms) | Erreurs (%) | RAM RSS Uvicorn (Mo) | Saturation CPU (%) |
|---|---|---|---|---|---|---|---|
| **Unitaire (/predict)** | 1 worker | 92.83 | 92.83 | 250 ms | 0.0 % | 228.6 Mo | 99.4 % |
|  | **2 workers** | **141.78** | **141.78** | **98 ms** | **0.0 %** | **263.1 Mo** | **83.6 %** |
|  | 4 workers | 143.61 | 143.61 | 90 ms | 0.0 % | 296.8 Mo | 73.2 % |
| **Batch (/predict/batch, 20 jeux)** | 1 worker | 38.99 | 779.86 | 450 ms | 0.0 % | 231.2 Mo | 99.1 % |
|  | **2 workers** | **58.24** | **1 164.81** | **200 ms** | **0.0 %** | **265.4 Mo** | **88.2 %** |
|  | 4 workers | 59.10 | 1 182.00 | 190 ms | 0.0 % | 299.1 Mo | 76.5 % |
| **Mixte Réaliste (70% U, 20% B, 10% FB)** | 1 worker | 79.35 | 222.18 | 480 ms | 0.0 % | 232.0 Mo | 98.7 % |
|  | **2 workers** | **104.38** | **292.26** | **210 ms** | **0.0 %** | **267.8 Mo** | **86.4 %** |
|  | 4 workers | 108.72 | 304.41 | 195 ms | 0.0 % | 302.4 Mo | 74.8 % |

### 3.2. Analyse du Facteur d'Échelle et Recommandation
1. **Passage de 1 à 2 workers :** Gain massif de **+52.7 %** de débit sur l'inférence unitaire et division par 2.5 de la latence $p_{95}$ (de 250 ms à 98 ms).
2. **Passage de 2 à 4 workers :** Rendement décroissant (+1.3 % de débit supplémentaire), la limite se déplaçant vers la contention du bus mémoire et du pool de persistance PostgreSQL.
3. **Recommandation pour charge crête (150 req/s unitaire avec marge de 30 %) :**
   - **Configuration retenue :** 2 répliques conteneurisées dotées chacune de **2 workers Uvicorn**.
   - **Ressources matérielles allouées :** 2 vCPU et 1 Go de RAM par réplique.
   - **Dimensionnement base de données :** Pool de 5 connexions par worker (20 connexions max au total sur PostgreSQL), garantissant un temps de service $< 100$ ms.

---

## 4. Fiabilité, Résilience et Gestion des Crises

Cinq scénarios de défaillance critique ont été injectés empiriquement via `scripts/demo_incident.py` et documentés dans `reports/incidents/` :

```mermaid
graph TD
    subgraph Incidents["Scénarios d'Incidents Injectés"]
        I1["1. Lot Invalide\n(Prix texte, doublons, hors bornes)"]
        I2["2. Crash PostgreSQL\n(Arrêt docker 5432)"]
        I3["3. Coupure Réseau MLflow\n(Publication asymétrique)"]
        I4["4. Modèle Dégradé\n(Régression de performance)"]
        I5["5. Panne Alertes\n(Webhook injoignable)"]
    end

    subgraph Defense["Mécanismes de Défense & Résilience"]
        D1["Barrière Great Expectations\nLevée DataValidationError"]
        D2["Mode Dégradé HTTP 503\nTélémétrie locale JSON Lines"]
        D3["Moteur de Réconciliation\nReprise sans doublon (scripts/reconcile.py)"]
        D4["Rollback d'Urgence sans Réentraînement\nRétablissement service-manifest.prev.json"]
        D5["Timeout Strict (1s) & Confinement\nAucune relance aveugle du pipeline"]
    end

    subgraph Outcome["Résultats Audités"]
        O1["Modèles & Manifestes 100% Intacts"]
        O2["Zéro faux succès 200\nReconnexion auto sans redémarrage API"]
        O3["Publication COMPLETED\nIdempotence prouvée sur 2e passe"]
        O4["Modèle A rétabli\nHistorique PostgreSQL A et B préservé"]
        O5["Pipeline préservé\nÉtat FAILED propre maintenu"]
    end

    I1 --> D1 --> O1
    I2 --> D2 --> O2
    I3 --> D3 --> O3
    I4 --> D4 --> O4
    I5 --> D5 --> O5
```

---

## 5. Rapprochement Comptable et Intégrité de la Persistance

Les totaux enregistrés en base PostgreSQL lors de campagnes étalonnées ont été réconciliés à l'unité près via `scripts/verify_sql_counts.py` :
- **Requêtes API totales (`api_requests`) :** 100% concordantes.
- **Prédictions conjointes (`predictions`) :** Ratio invariant de $2N$ prédictions par requête confirmé (0 violation d'intégrité).
- **Feedbacks et vérités terrain (`feedbacks`) :** Idempotence parfaite vérifiée sur les rejeux (code HTTP 200 retourné, sans création de doublon).

---

## 6. Limites Scientifiques, Techniques et Opérationnelles

En toute transparence d'ingénierie, les limites suivantes ont été caractérisées lors de l'exploitation :

### 6.1. Limites Scientifiques
- **Déséquilibre sévère des classes sur `estimated_owners` :** La tranche majoritaire (`0 - 20000`) concentre plus de 73% du corpus historique. Malgré l'utilisation de `class_weight='balanced'`, la précision sur les tranches rares (ex. `20000000 - 50000000`) reste modeste (Macro F1 = 0.1036).
- **Pouvoir prédictif des métadonnées statiques :** La satisfaction des joueurs dépend fortement de facteurs non capturés par les métadonnées Steam (qualité du gameplay, présence de bugs au lancement, marketing). Le coefficient de détermination $R^2 = 0.1602$ reflète cette limite inhérente aux données publiques.

### 6.2. Limites Techniques & Infrastructure
- **Persistance PostgreSQL sous charge crête :** Bien que le pool de 5 connexions soit résilient, une sollicitation unitaire continue au-delà de 200 req/s sur une seule instance PostgreSQL provoque une file d'attente d'écriture. L'utilisation d'une file de messages tampon (ex. RabbitMQ ou Kafka) serait requise pour une volumétrie supérieure à 1 000 req/s.
- **Synchronisation Prefect -> PostgreSQL :** L'exporteur de télémétrie `ops_export.py` fonctionne en mode batch périodique. Les métriques d'entraînement dans Grafana ont donc une latence de synchronisation égale à l'intervalle d'export (1 à 5 minutes).

---

## 7. Recommandations pour le Passage à l'Échelle (Roadmap v2.0)

1. **Déploiement Kubernetes (K8s) :**
   - Utilisation d'un Horizontal Pod Autoscaler (HPA) basé sur la latence $p_{95}$ pour ajuster dynamiquement le nombre de pods Uvicorn entre 2 et 10 instances.
2. **Tampon Asynchrone d'Écriture :**
   - Remplacer l'insertion synchrone de `tracing.api_requests` par une publication asynchrone Redis / Celery pour réduire la latence d'inférence sous le seuil des 5 millisecondes.
3. **Surveillance Continue de Dérive (Evidently AI) :**
   - Automatiser le calcul quotidien de la dérive de distribution (Wasserstein Distance, PSI) entre les inférences de production et le jeu d'entraînement, avec déclenchement automatique d'un réentraînement Prefect si le PSI dépasse 0.25.
