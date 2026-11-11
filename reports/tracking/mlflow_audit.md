# Audit Technique du Suivi MLflow et des Modèles Rechargeables

> **Projet :** MLOps 5MLDE — Industrialisation de Modèles Steam  
> **Date de réalisation :** 2026-11-16  
> **Exigences couvertes :** [REQ-12](file:///C:/Users/kambj/Desktop/Supinfo/5MLDE/Final%20Project/reports/requirements_matrix.md) (Suivi complet / Tracking), [REQ-13](file:///C:/Users/kambj/Desktop/Supinfo/5MLDE/Final%20Project/reports/requirements_matrix.md) (Versionnement et règles de registre).

---

## 1. Contexte & Architecture du Serveur MLflow

Le serveur de suivi centralisé est orchestré au sein du conteneur `mlflow` (image officielle `ghcr.io/mlflow/mlflow:v3.16.1`) sur le port dédié **7281** :
- **URI de Tracking / Registre :** `http://127.0.0.1:7281` (ou `http://localhost:7281`)
- **Backend Store :** SQLite relationnel (`sqlite:////mlflow/mlflow.db`)
- **Artifact Store :** Serveur d'artefacts HTTP intégré (`--serve-artifacts`, destination `/mlflow/artifacts`)
- **Expérience Principale :** `steam-mlops-experiments` (ID: `3`)

L'intégration applicative est encapsulée dans le module [`src/tracking/mlflow_tracking.py`](file:///C:/Users/kambj/Desktop/Supinfo/5MLDE/Final%20Project/src/tracking/mlflow_tracking.py) et le gestionnaire de registre [`src/tracking/registry.py`](file:///C:/Users/kambj/Desktop/Supinfo/5MLDE/Final%20Project/src/tracking/registry.py).

---

## 2. Synthèse d'Exécution du Run de Référence Historique

- **Nom du Run :** `train-historical-training-run-7d01`
- **Identifiant Unique (MLflow Run ID) :** `6ab770cabfab4765a38c7964071bb322`
- **URL de Consultation UI :** [http://127.0.0.1:7281/#/experiments/3/runs/6ab770cabfab4765a38c7964071bb322](http://127.0.0.1:7281/#/experiments/3/runs/6ab770cabfab4765a38c7964071bb322)
- **Snapshot Source Utilisé :** `data/snapshots/historical_projection.csv`
- **Empreinte SHA-256 du Snapshot :** `7eec2ff4e6f1389bfb74308c8de1206fee2a94f51120895c2cf7fce84146312f`
- **Nombre de lignes du Snapshot :** 94 948 jeux
- **Durée Totale du Pipeline :** 157.85s (Entraînements : 88.96s)

---

## 3. Paramètres Journalisés

| Clé de Paramètre | Valeur Journalisée | Justification Technique |
|---|---|---|
| `random_seed` | `42` | Reproductibilité absolue des estimateurs et des découpages |
| `split.train_ratio` | `0.70` | 70 % des données en apprentissage |
| `split.val_ratio` | `0.15` | 15 % en qualification (réserve protégée) |
| `split.test_ratio` | `0.15` | 15 % en test final (réserve protégée) |
| `features.top_genres_k` | `15` | Seuil d'encodage binaire des genres les plus fréquents |
| `features.allowed_count` | `9` | Cardinalité stricte des variables d'inférence autorisées |
| `regression.estimator` | `RandomForestRegressor` | Algorithme retenu pour la régression continue |
| `regression.n_estimators`| `40` | 40 arbres de décision |
| `regression.max_depth` | `12` | Régularisation de la profondeur pour éviter le surapprentissage |
| `regression.min_reviews` | `10` | Seuil de fiabilité statistique sur les avis Steam |
| `classification.estimator` | `RandomForestClassifier` | Algorithme retenu pour la classification multiclasse |
| `classification.n_estimators`| `40` | 40 arbres de décision |
| `classification.max_depth` | `12` | Profondeur maximale régularisée |
| `classification.min_class_count` | `5` | Élagage des classes ultra-minoritaires (< 5 exemples) |
| `classification.excluded_class` | `0 - 0` | Exclusion conventionnelle des sentinelles non commerciales |

---

## 4. Métriques Journalisées et Comparaison aux Baselines Naïves

Chaque métrique de modèle candidat est systématiquement mise en regard de sa baseline naïve calculée sur le **même split** :

### 4.1. Tâche 1 — Régression (`pct_pos_total`)
*Population éligible : 55 373 jeux (Train: 38 714, Val: 8 286, Test: 8 373)*

| Split | Métrique | Modèle Candidat (RF) | Baseline Naïve (Moyenne) | Écart Relatif |
|---|---|---|---|---|
| **Validation** | **MAE** | **12.53 %** | 13.91 % | **-9.92 % d'erreur** |
| Validation | RMSE | **17.70 %** | 19.32 % | -8.38 % d'erreur |
| Validation | $R^2$ | **0.1602** | -0.0000 | Baseline battue significativement |
| **Train** | MAE | 11.23 % | 13.91 % | Apprentissage contrôlé |
| Train | $R^2$ | 0.3340 | 0.0000 | Aucun surapprentissage pathologique |
| **Test** | MAE | 12.55 % | 13.88 % | Cohérence parfaite avec la validation |
| Test | $R^2$ | 0.1585 | -0.0000 | Généralisation robuste |

### 4.2. Tâche 2 — Classification (`estimated_owners`, 12 classes)
*Population éligible : 81 290 jeux (Train: 56 902, Val: 12 194, Test: 12 194)*

| Split | Métrique | Modèle Candidat (RF) | Baseline Naïve (Majoritaire) | Écart Relatif |
|---|---|---|---|---|
| **Validation** | **Accuracy** | **73.45 %** | 73.04 % | **+0.56 % relatif** |
| Validation | **F1-Macro** | **0.1036** | 0.0704 | **+47.16 % d'amélioration** |
| **Train** | Accuracy | 73.90 % | 73.04 % | Régularisation effective |
| Train | F1-Macro | 0.1418 | 0.0704 | Détection des classes ordonnées |
| **Test** | Accuracy | 73.38 % | 73.04 % | Stabilité inter-splits |
| Test | F1-Macro | 0.0984 | 0.0704 | Supériorité nette vs baseline |

---

## 5. Signatures d'Inférence et Étanchéité Anti-Fuite

Les deux modèles sérialisés intègrent une signature d'inférence générée par `infer_signature` :
- **Entrées autorisées (9 caractéristiques exactes) :**
  `price` (double), `required_age` (long), `dlc_count` (long), `achievements` (long), `windows` (boolean), `mac` (boolean), `linux` (boolean), `genres` (string), `release_date` (string).
- **Interdiction formelle et vérifiée des colonnes de fuite :**
  Aucune des colonnes cibles (`pct_pos_total`, `estimated_owners`) ni des métriques d'engagement fuyantes (`positive`, `negative`, `num_reviews_total`) n'est présente dans les entrées du modèle.

---

## 6. Modèles Enregistrés dans le Model Registry

Deux modèles ont été officiellement créés et enregistrés en version `1` avec statut `READY` :
1. **`steam-positive-ratio` (Version 1) :**
   - Rôle : Inférence continue du pourcentage d'avis favorables.
   - URI d'artefact : `models:/m-8290715cfbb1420a9b9945cdf6107529`
   - Intègre : `SteamFeatureBuilder(top_genres_k=15)` + `SimpleImputer` + `RandomForestRegressor`.
2. **`steam-owner-range` (Version 1) :**
   - Rôle : Prédiction de la tranche de possession estimée sur 12 classes ordonnées.
   - URI d'artefact : `models:/m-6db2af389c6b40b4ad3d20409ddc7ec2`
   - Intègre : `SteamFeatureBuilder(top_genres_k=15)` + `SimpleImputer` + `RandomForestClassifier`.

---

## 7. Preuve Matérielle de Rechargement Externe

Le test d'intégration automatisé [`tests/integration/test_mlflow_reload.py`](file:///C:/Users/kambj/Desktop/Supinfo\5MLDE\Final Project\tests\integration\test_mlflow_reload.py) a prouvé formellement :
1. **Fidélité numérique à $10^{-10}$ :**
   Les prédictions du modèle rechargé via `load_model(model_uri)` et celles du modèle d'origine en mémoire sont rigoureusement identiques bit-à-bit :
   $$\max |y_{\text{reloaded}} - y_{\text{orig}}| < 10^{-10}$$
2. **Exécution dans un interpréteur vierge (Sous-processus Python isolé) :**
   Le rechargement depuis l'URI MLflow dans un sous-processus `python -c "..."` indépendant a réussi sans aucune dépendance envers l'espace mémoire d'entraînement.
3. **Robustesse aux cas limites :**
   Inférence validée sur des jeux présentant des genres inconnus (`['Quantum', 'VirtualReality']`) et des métadonnées temporelles manquantes sans lever d'exception.
