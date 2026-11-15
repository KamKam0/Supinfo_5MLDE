# Contrats d'Interface et Spécification de l'API de Prédiction (docs/api_contract.md)

> **Projet :** Industrialisation de Modèles Steam — MLOps 5MLDE  

L'API de prédiction est exposée sur le port **8001** via FastAPI et Uvicorn. Elle fournit les services d'inférence en ligne haute performance et d'enregistrement transactionnel des retours pour le catalogue Steam.

---

## 1. Vue d'Ensemble des Endpoints

| Méthode | Chemin | Rôle & Fonctionnalité | Code Nominal | Codes d'Erreur |
|---|---|---|---|---|
| `POST` | `/predict` | Inférence conjointe unitaire avec persistance SQL atomique | `200 OK` | `422`, `503` |
| `POST` | `/predict/batch` | Inférence vectorisée par lot (1 à 100 jeux) et persistance des $2N$ prédictions | `200 OK` | `422`, `503` |
| `POST` | `/feedback` | Réception transactionnelle des retours utilisateurs et vérités terrain | `201 Created` / `200 OK` | `404`, `409`, `422`, `503` |
| `GET` | `/health/live` | Sonde de vivacité confirmant que le processus répond | `200 OK` | `503` |
| `GET` | `/health/ready` | Sonde d'aptitude (modèles en mémoire ET connectivité PostgreSQL) | `200 OK` | `503` |
| `GET` | `/models` | Consultation du manifeste actif et des versions de modèles figées | `200 OK` | `503` |

---

## 2. Contrats de Données & Schémas Pydantic

### 2.1. Schéma d'Entrée d'un Jeu (`InferenceGameInput`)

Ce schéma régit la structure stricte d'un jeu vidéo soumis à l'inférence.  
**RÈGLE D'OR ABSOLUE (Anti-fuite) :** La configuration `extra = "forbid"` rejette formellement en **HTTP 422** toute requête contenant une variable cible (`pct_pos_total`, `estimated_owners`, `positive`, `negative`, `num_reviews_total`).

```json
{
  "appid": 1086940,
  "name": "Baldur's Gate 3",
  "price": 59.99,
  "required_age": 18,
  "genres": ["RPG", "Strategy", "Adventure"],
  "windows": true,
  "mac": true,
  "linux": false,
  "release_date": "2023-08-03",
  "dlc_count": 0,
  "achievements": 53
}
```

| Champ | Type | Contraintes | Description |
|---|---|---|---|
| `appid` | Integer | `> 0`, obligatoire | Identifiant unique Steam du jeu |
| `name` | String | Obligatoire | Titre officiel du jeu |
| `price` | Float | `price >= 0.0` | Prix public en USD |
| `required_age` | Integer | `0 <= age <= 100` | Âge minimal requis |
| `genres` | List[String] | Optionnel (défaut `[]`) | Liste des genres officiels Steam |
| `windows` | Boolean | Optionnel (défaut `true`) | Compatibilité Windows |
| `mac` | Boolean | Optionnel (défaut `false`) | Compatibilité macOS |
| `linux` | Boolean | Optionnel (défaut `false`) | Compatibilité Linux |
| `release_date` | String | Format ISO `YYYY-MM-DD` | Date de sortie commerciale |
| `dlc_count` | Integer | `dlc_count >= 0` | Nombre de contenus additionnels |
| `achievements` | Integer | `achievements >= 0` | Nombre de succès déblocables |

---

### 2.2. Réponse d'Inférence Conjointe (`PredictionResponse`)

Chaque prédiction délivre **conjointement** les estimations des deux modèles :
1. `predicted_pct` : Pourcentage d'avis positifs estimé ($\in [0.0, 100.0]$).
2. `predicted_owners` : Tranche de propriétaires estimée (sur 12 classes).

```json
{
  "request_id": "req-98b7f3a1-c24e-4b67-91a0-9742c3882741",
  "manifest_id": "man-a7f270fb",
  "item_count": 1,
  "results": [
    {
      "item_index": 0,
      "predicted_pct": 84.12,
      "predicted_owners": "2000000 - 5000000",
      "predictions": [
        {
          "task": "regression",
          "model_name": "steam-positive-ratio",
          "model_version": "12",
          "prediction_id": "89b7f3a1-c24e-4b67-91a0-9742c3882741",
          "predicted_value": 84.12
        },
        {
          "task": "classification",
          "model_name": "steam-owner-range",
          "model_version": "12",
          "prediction_id": "99b7f3a1-c24e-4b67-91a0-9742c3882742",
          "predicted_label": "2000000 - 5000000"
        }
      ]
    }
  ],
  "processing_duration_ms": 14.52,
  "inference_duration_ms": 3.84
}
```

---

### 2.3. Réception des Feedbacks (`POST /feedback`)

L'endpoint `/feedback` supporte deux cas d'usage distincts :

#### Cas A : Observation Réelle (Vérité Terrain / Ground Truth)
Soumission d'une valeur observée réelle (`observed_pct` pour une tâche de régression, ou `observed_owners` pour une classification).
```json
{
  "feedback_id": "fb-89b7f3a1-0001",
  "prediction_id": "89b7f3a1-c24e-4b67-91a0-9742c3882741",
  "observed_pct": 82.5,
  "label_source": "manual_review",
  "comment": "Revue manuelle des statistiques réelles après campagne"
}
```

#### Cas B : Avis d'Utilité Subjectif Pur (Sans Vérité Terrain)
**RÈGLE D'OR :** Un avis subjectif pur (`useful = true / false`) sans observation réelle **interdit formellement** la présence du champ `label_source`. Tout payload combinant `useful` seul avec `label_source` est rejeté en **HTTP 422**.
```json
{
  "feedback_id": "fb-89b7f3a1-0002",
  "prediction_id": "99b7f3a1-c24e-4b67-91a0-9742c3882742",
  "useful": true,
  "comment": "Prédiction utile pour le dimensionnement du serveur"
}
```

#### Gestion des Codes Retours sur `/feedback` :
- `201 Created` : Nouveau feedback enregistré avec succès.
- `200 OK` : **Rejeu idempotent** strict (même `feedback_id`, même `prediction_id` et même contenu).
- `404 Not Found` : `prediction_id` inexistant dans la base PostgreSQL `tracing.predictions`.
- `409 Conflict` : Conflit de feedback (même `feedback_id` avec un contenu différent, ou prédiction déjà annotée).
- `422 Unprocessable Content` : Incohérence métier (ex. observation `observed_pct` envoyée sur une classification, ou présence interdite de `label_source`).
- `503 Service Unavailable` : Base PostgreSQL indisponible (erreur consignée dans `state/telemetry_fallback.jsonl`).

---

## 3. En-têtes HTTP de Corrélation et d'Audit

Chaque réponse HTTP émise par l'API de prédiction comporte les en-têtes d'audit immuables suivants :

| En-tête HTTP | Description & Format | Exemple |
|---|---|---|
| `X-Request-ID` | Identifiant UUID de corrélation universel | `89b7f3a1-c24e-4b67-91a0-9742c3882741` |
| `X-Manifest-ID` | Identifiant du manifeste de service actif | `man-a7f270fb` |
| `X-Worker-Instance-ID` | UUID unique de l'instance Uvicorn ayant traité l'appel | `4ddf8529-57f9-454a-9f91-2b7438d92844` |
| `X-Model-Regression-Version` | Numéro de version exact du modèle de régression | `12` |
| `X-Model-Classification-Version` | Numéro de version exact du modèle de classification | `12` |
| `X-Test-Campaign` | (Optionnel, entrant) Identifiant de campagne de charge | `loadtest-w2-batch-u25` |

---

## 4. Spécification OpenAPI et Documentation Interactive

- **Spécification JSON brute :** [`docs/openapi.json`](file:///C:/Users/kambj/Desktop/Supinfo/5MLDE/Final%20Project/docs/openapi.json)
- **Documentation Swagger UI :** Accessible à l'adresse `http://127.0.0.1:8001/docs`
- **Documentation Redoc :** Accessible à l'adresse `http://127.0.0.1:8001/redoc`
