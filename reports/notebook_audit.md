# Audit Critique du Notebook Exploratoire & Plan de Refactoring

Ce document analyse les limitations, risques d'intégrité et faiblesses architecturales du notebook initial (`exploration_steam.ipynb`) du parcours pédagogique, et détaille les remédiations apportées par l'architecture logicielle modulaire du socle MLOps.

---

## 1. Matrice des Défauts Identifiés et Corrections MLOps

| # | Défaut Constaté dans le Notebook | Risque / Effet Négatif en Production | Solution Technique Implémentée (Socle MLOps) |
|---|---|---|---|
| **1** | **Variables globales partagées** (`X`, `y`, `model`) | Écrasement silencieux d'une variable par une autre, confusion entre les métriques de régression et de classification. | Structures de données typées et cloisonnées (`TaskDataset`, `PreparedData`) dans [`src/contracts.py`](file:///C:/Users/kambj/Desktop/Supinfo/5MLDE/Final%20Project/src/contracts.py). |
| **2** | **Réutilisation du même nom de modèle** | Impossibilité de sérialiser ou déployer simultanément les deux tâches sans écraser le fichier de poids. | Deux pipelines scikit-learn distincts avec leurs propres estimateurs et artefacts (`artifacts/regression.joblib` et `artifacts/classification.joblib`). |
| **3** | **Prétraitement dupliqué et incohérent** | Risque élevé de divergence entre le code d'entraînement et le code de prédiction en cas d'évolution. | Transformeur unique et standardisé [`SteamFeatureBuilder`](file:///C:/Users/kambj/Desktop/Supinfo/5MLDE/Final%20Project/src/feature_builder.py) respectant l'API scikit-learn (`fit`/`transform`). |
| **4** | **Splits aléatoires sans graine explicite** | Résultats non reproductibles d'une exécution à l'autre, impossibilité de comparer objectivement des modèles. | Partitionnement déterministe (`random_state=42`) et figeage strict des identifiants dans [`data/partitions/*.json`](file:///C:/Users/kambj/Desktop/Supinfo/5MLDE/Final%20Project/data/partitions/). |
| **5** | **Chemins relatifs au répertoire courant** | Échec à l'exécution dès que le script est lancé depuis un autre sous-dossier ou en tâche planifiée. | Chemins résolus de manière absolue à partir de la racine du projet ou configurés via [`src/settings.py`](file:///C:/Users/kambj/Desktop/Supinfo/5MLDE/Final%20Project/src/settings.py). |
| **6** | **Dépendance à l'état du kernel Jupyter** | Résultats dépendants de l'ordre manuel d'exécution des cellules, non-reproductibilité en pipeline batch. | Suppression définitive de tout notebook dans la chaîne d'exécution ; orchestration par scripts CLI purs (`main.py`, `predict.py`). |
| **7** | **Sauvegarde immédiate sans contrôle qualité** | Écrasement d'un modèle en production par un modèle corrompu ou régressif en cas d'erreur en fin de run. | Publication après validation complète et comparaison formelle aux baselines naïves. |
| **8** | **Contrats d'interface implicites** | Entrées incohérentes entre l'entraînement et l'inférence, risque majeur de fuite de variables cibles (*data leakage*). | Contrats Pydantic stricts ([`src/contracts.py`](file:///C:/Users/kambj/Desktop/Supinfo/5MLDE/Final%20Project/src/contracts.py)) avec `extra="forbid"` interdisant les cibles à l'inférence. |
| **9** | **Fuite dans l'imputation des données** | Calcul des moyennes/médianes sur l'ensemble du dataset avant découpage Train/Test, biaisant les métriques. | Ajustement (`fit`) des imputeurs et encodeurs **strictement sur la partition Train**, puis simple propagation (`transform`) sur Val et Test. |

---

## 2. Décomposition en Quatre Fonctions Mères

Pour garantir une modularité totale et une testabilité unitaire sans effet de bord, le pipeline est refactorisé selon 4 fonctions mères aux responsabilités étanches :

1. **`load_data(config_path)` (`src/ingestion.py`) :**
   - Charge la configuration résolue et le snapshot de données immuable.
   - Contrôle l'empreinte cryptographique SHA-256.
   - Prépare le contexte d'exécution (`RunContext`, `SnapshotRef`).

2. **`preprocess_data(ingestion_result)` (`src/preprocessing.py`) :**
   - Applique les filtres d'éligibilité métier pour la régression et la classification.
   - Sépare les sous-ensembles Train, Validation et Test sur la base des partitions figées.
   - Isole les matrices de caractéristiques et les vecteurs cibles sans fuite de données.

3. **`train_models(prepared_data)` (`src/training.py`) :**
   - Instancie les pipelines scikit-learn complets (`SteamFeatureBuilder` + imputeur + estimateur).
   - Entraîne les modèles candidats (`RandomForest`) et les baselines naïves associées (`DummyRegressor`, `DummyClassifier`).
   - Mesure les temps d'apprentissage.

4. **`evaluate_models(training_result, prepared_data)` (`src/evaluation.py`) :**
   - Calcule les métriques d'évaluation sur Train, Validation et Test (MAE, RMSE, $R^2$, Accuracy, F1-macro).
   - Compare systématiquement les performances aux baselines.
   - Sérialise les modèles et sauvegarde les métriques dans `artifacts/`.
