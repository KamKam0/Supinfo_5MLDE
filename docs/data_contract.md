# Contrat de Données, Règles Métier et Politique Anti-Fuite (Data Contract)

Ce document établit la spécification formelle des données traitées, les contrats d'interface Pydantic, les règles d'éligibilité pour les deux tâches d'apprentissage et le protocole mathématique anti-fuite (*data leakage*).

---

## 1. Dictionnaire des Données et Types Logiques

La table suivante récapitule les 16 champs de la projection canonique (`data/snapshots/historical_projection.csv`) :

| Champ | Type Python | Usage MLOps | Traitement des Sentinelles & Règles |
|---|---|---|---|
| `appid` | `int` | Identifiant & Audit | Strictement unique ($> 0$), **exclu des features d'apprentissage**. |
| `name` | `str` | Audit & Affichage | Présent pour l'humain, **exclu des features d'apprentissage**. |
| `release_date` | `str` | Feature temporelle | Format ISO ou textuel ; extraction de l'année de sortie (`release_year`). |
| `price` | `float` | Feature numérique | Prix standard $\ge 0.0$. Zéro correspond à un jeu *Free to Play*. |
| `required_age` | `int` | Feature numérique | Âge minimal légal. `-1` est une sentinelle historique convertie en null/médiane. |
| `dlc_count` | `int` | Feature numérique | Comptage $\ge 0$. |
| `achievements` | `int` | Feature numérique | Nombre de succès déblocables $\ge 0$. |
| `windows` | `bool` | Feature booléenne | Compatibilité OS (`True` / `False`). |
| `mac` | `bool` | Feature booléenne | Compatibilité OS (`True` / `False`). |
| `linux` | `bool` | Feature booléenne | Compatibilité OS (`True` / `False`). |
| `genres` | `str` / `list[str]` | Feature catégorielle | Liste des genres ; extraction du genre dominant (`primary_genre`). |
| `positive` | `int` | Exploration uniquement | Compteur brut d'avis favorables. **Interdit dans les features d'inférence (fuite)**. |
| `negative` | `int` | Exploration uniquement | Compteur brut d'avis défavorables. **Interdit dans les features d'inférence (fuite)**. |
| `pct_pos_total` | `float` | **Cible Régression** | Pourcentage d'avis positifs dans $[0.0, 100.0]$. `-1.0` signale l'absence d'évaluation. |
| `num_reviews_total` | `int` | Filtre d'éligibilité | Seuil minimal de fiabilité statistique. **Interdit dans les features (fuite)**. |
| `estimated_owners` | `str` | **Cible Classification** | Tranche textuelle (ex. `'20000 - 50000'`). `0 - 0` exclu conventionnellement. |

---

## 2. Règles d'Éligibilité par Tâche Prédictive

Chaque modèle opère sur une sous-population rigoureusement définie :

### 2.1. Tâche de Régression (`pct_pos_total`)
- **Critère de validité :** `pct_pos_total != -1` ET `num_reviews_total >= 10`.
- **Règle d'or :** Ne jamais imputer une cible manquante par zéro. Les jeux ayant moins de 10 avis ne possèdent pas de signal statistique fiable et sont exclus de l'entraînement et de l'évaluation de régression.
- **Effectif éligible :** **55 373 jeux** sur les 94 948 du snapshot historique.

### 2.2. Tâche de Classification (`estimated_owners`)
- **Critère de validité :** `estimated_owners != '0 - 0'` ET effectif de classe $\ge 5$.
- **Règle d'or :** Les 13 656 jeux portant la tranche conventionnelle `0 - 0` sont exclus. Les classes comptant moins de 5 jeux (`100000000 - 200000000` et `200000000 - 500000000`, 1 jeu chacune) sont hors périmètre pour assurer un apprentissage stable.
- **Effectif éligible :** **81 290 jeux** répartis sur **12 classes**.

---

## 3. Contrat d'Inférence Anti-Fuite (`InferenceInputGame`)

L'API de prédiction applique une barrière d'étanchéité stricte via Pydantic (`extra = 'forbid'`) :

- **Attributs acceptés :**
  - `appid`, `price`, `required_age`, `dlc_count`, `achievements`, `windows`, `mac`, `linux`, `genres`, `release_date`, `name` (optionnel).
- **Attributs formellement rejetés (Erreur HTTP 422) :**
  - `pct_pos_total`, `estimated_owners`, `num_reviews_total`, `positive`, `negative`.
- Tout champ surnuméraire entraîne un rejet immédiat de la requête avant tout appel au modèle.

---

## 4. Politique de Partitionnement et Contrôle Anti-Fuite

### 4.1. Répartition Déterministe
Pour éviter toute fuite d'information temporelle ou structurelle, les partitions sont fixées au niveau des identifiants `appid` uniques avec une stratification sur les classes d'`estimated_owners` :

- **Sous-ensemble Train (70 %) :** 66 463 appids (`data/partitions/train_appids.json`).
- **Sous-ensemble Validation commune (15 %) :** 14 242 appids (`data/partitions/val_appids.json`).
- **Sous-ensemble Test indépendant (15 %) :** 14 243 appids (`data/partitions/test_appids.json`).
- **Réserve Globale Protégée (30 %) :** 28 485 appids (`data/partitions/reserved_appids.json`).

### 4.2. Règle Absolue d'Étanchéité
Les partitions satisfont rigoureusement les propriétés ensemblistes :
$$\text{Train} \cap \text{Validation} = \emptyset \quad \text{et} \quad \text{Train} \cap \text{Test} = \emptyset \quad \text{et} \quad \text{Validation} \cap \text{Test} = \emptyset$$

### 4.3. Protection des Runs Futurs (PULL / PUSH)
La fonction `filter_out_reserved_games(df, reserved_ids)` de `src/partitions.py` est systématiquement appelée lors de l'ingestion de tout nouveau lot. Aucun `appid` appartenant à `data/partitions/reserved_appids.json` ne pourra être injecté dans un lot d'apprentissage présent ou futur.
