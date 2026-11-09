# Matrice de Correspondance des Règles de Qualité des Données (Great Expectations)

> **Document de Référence :** `reports/validation/validation_rules_matrix.md`  
> **Cadre :** Barrière de qualité bloquante  
> **Moteur :** Great Expectations Core 1.23.2 & Contrôles Métier Déterministes  

---

## 1. Vue d'Ensemble des 5 Suites de Validation

| Nom de la Suite | Périmètre d'Application | Objectif & Moment d'Exécution | Action en Cas d'Échec |
|---|---|---|---|
| `historical_schema_v1` | Snapshot historique projeté (`94 948` lignes) | Fin de `load_data()`, contrôle des 16 colonnes et sentinelles documentées | Code retour non nul, arrêt immédiat du pipeline |
| `new_observations_v1` | Nouvelles collectes brutes (PULL API ou PUSH) | Avant toute fusion (*upsert*) avec le catalogue existant | Rejet du lot, aucune altération du snapshot |
| `snapshot_integrity_v1` | Snapshot candidat consolidé | Post-upsert, avant qualification pour réentraînement | Révocation du snapshot candidat |
| `training_readiness_v1` | Sous-ensembles d'apprentissage stratifiés | Avant découpage et `preprocess_data()` | Interdiction d'entraîner, conservation des modèles existants |
| `push_batch_v1` | Lot PUSH tamponné (`exactement 500 jeux`) | Réception du buffer PUSH lors du déclenchement | Rejet du buffer, passage du job à `FAILED` |

---

## 2. Matrice des 10 Familles de Règles

| ID Règle | Règle Métier | Expectation GX / Contrôle Métier | Suites Actives | Traitement des Nulls & Sentinelles | Conduite en Échec |
|---|---|---|---|---|---|
| **R01-SCHEMA** | Présence des 16 colonnes obligatoires | `ExpectTableColumnsToMatchSet(column_set=..., exact_match=False)` | Toutes | Nulls interdits sur la structure | `DataSchemaError` immédiate |
| **R02-VOLUME** | Volume minimal ou exact du jeu de données | `ExpectTableRowCountToBeBetween` / `ExpectTableRowCountToEqual` | Toutes (`push_batch_v1` = 500) | Tableau non vide requis | `DataValidationError` |
| **R03-APPID** | Identifiant unique, entier strictement positif | `ExpectColumnValuesToNotBeNull`, `ExpectColumnValuesToBeUnique`, `ExpectColumnValuesToBeBetween(min_value=1)` | Toutes | Aucun null toléré, strictement $> 0$ | `DataIntegrityError` |
| **R04-NUMERIC** | Prix et compteurs finis et non négatifs | `ExpectColumnValuesToBeBetween(min_value=0.0)` pour `price`, `dlc_count`, `achievements`, `positive`, `negative` | Toutes | `price` $\ge 0$, `dlc` $\ge 0$, `achievements` $\ge 0$ | `DataValidationError` |
| **R05-SENTINEL-AGE** | Gestion de la sentinelle d'âge minimal | `ExpectColumnValuesToBeBetween(column="required_age", ...)` | Hist: $[-1, 120]$<br>Nouv: $[0, 120]$ | `-1` autorisé uniquement dans l'historique brut | `DataValidationError` |
| **R06-SENTINEL-REVIEWS** | Gestion du statut d'avis insuffisants | `ExpectColumnValuesToBeBetween(column="num_reviews_total", ...)` | Hist: $[-1, \infty[$<br>Nouv: $[0, \infty[$ | `-1` autorisé dans l'historique quand `num_reviews_total < 10` | `DataValidationError` |
| **R07-TARGET-PCT** | Bornes du pourcentage d'avis positifs | `ExpectColumnValuesToBeBetween(column="pct_pos_total", ...)` | Hist: $[-1.0, 100.0]$<br>Nouv: $[0.0, 100.0]$ | `-1` exclu formellement de toute nouvelle observation | `DataValidationError` |
| **R08-TARGET-OWNERS** | Format et ordre des tranches de possession | Regex `^\d+ - \d+$` + Contrôle $low < high$ | Toutes (Hist tolère `'0 - 0'`) | `'0 - 0'` banni des flux nouveaux et du PUSH | `DataValidationError` |
| **R09-PLATFORMS-DATES** | Plateformes booléennes et date ISO | `ExpectColumnValuesToBeInSet` sur booléens, Regex `^\d{4}-\d{2}-\d{2}$` sur `release_date` | Toutes | Types booléens stricts, dates valides | `DataValidationError` |
| **R10-ANTI-LEAKAGE** | Étanchéité absolue : aucun appid réservé en Train | Contrôle d'intersection ensembliste avec `data/partitions/reserved_appids.json` | `training_readiness_v1` | $\text{Train} \cap \text{Reserved} = \emptyset$ obligatoire | `DataLeakageError` |

---

## 3. Positionnement Architectural de la Barrière

```mermaid
flowchart TD
    A["Fichier source / Snapshot CSV"] --> B["load_data() (src/ingestion.py)"]
    B --> C["Barrière Great Expectations (src/data_validation.py)"]
    C -->|Succès (Validation OK)| D["IngestionResult(gx_success=True)"]
    C -->|Échec / Anomalie détectée| E["Rapport archivé dans reports/validation/"]
    E --> F["Levée immédiate de DataValidationError"]
    F --> G["Arrêt Processus (Code != 0)"]
    G --> H["Anciens modèles conservés intacts"]
    D --> I["preprocess_data() (src/preprocessing.py)"]
    I --> J["train_models() (src/training.py)"]
    J --> K["evaluate_models() (src/evaluation.py)"]
```

---

## 4. Politique de Diagnostic et Archivage

Lors de chaque validation :
1. Un rapport JSON synthétique est écrit dans `reports/validation/latest_validation_report.json`.
2. En cas d'échec :
   - Un rapport d'audit permanent et horodaté est produit : `reports/validation/validation_failure_<suite>_<timestamp>.json`.
   - Les informations recueillies incluent :
     - La suite de validation et le jeu analysé.
     - Le nombre total d'anomalies.
     - Les détails de chaque attente non respectée (attribut `result` de GX).
     - La liste échantillon des identifiants `appid` en infraction (jusqu'à 20).
     - Le chemin absolu du rapport d'audit.
   - Les pipelines avals (`preprocess_data`, `train_models`) sont protégés par conception : toute tentative d'exécution en état d'échec est neutralisée.
