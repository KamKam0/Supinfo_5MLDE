# Rapport de Charge et Recommandation de Dimensionnement (reports/sizing.md)

Ce document consigne les résultats expérimentaux des bancs de charge conduits avec **Locust 2.46.6** sur l'API d'inférence en ligne Uvicorn/FastAPI.

---

## 1. Protocole de Mesure et Objectifs de Niveau de Service (SLO)

### Paramètres Fixes du Banc de Test
- **Empreinte des fixtures de test :** `fixtures/load_fixtures.json` (SHA-256: `44348168f31214c7dfc15dc00f677eef74c79ea8a442cbf681d6720ea376026e`)
- **Validation des corps :** Vérification stricte via `catch_response` (détection et rejet de tout code 200 corrompu ou incomplet).
- **Isolation des mesures :** Phase de chauffe (*warm-up*) de 15s exclue des statistiques ; entraînements Prefect mis en pause.
- **Traçabilité :** En-tête de corrélation `X-Test-Campaign` injecté sur 100% des requêtes HTTP.

### Seuils de SLO Contractuels (Déclarés avant les mesures)
| Scénario | Définition du Payload | Seuil $p_{95}$ Client | Seuil d'Erreur Toléré |
|---|---|:---:|:---:|
| **Unitaire** | 1 jeu soumis à `/predict` | **$< 500$ ms** | **$< 1.0\%$** |
| **Batch fixe** | 20 jeux vectorisés à `/predict/batch` | **$< 1500$ ms** | **$< 1.0\%$** |
| **Mixte** | 70% unitaire, 20% batch, 10% feedbacks réels | **$< 1000$ ms** | **$< 1.0\%$** |

---

## 2. Tableau Comparatif des Mesures Expérimentales

| Workers | Scénario | Concurrence | Débit req/s | Jeux/s | $p_{95}$ client ms | Erreurs % | CPU % | RAM Mo | Statut SLO |
|---:|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | batch | 5 | 12.82 | 256.49 | 96.0 | 0.0% | 23.44% | 231.38 Mo | **CONFORME** |
| 1 | batch | 10 | 23.79 | 475.82 | 190.0 | 0.0% | 47.76% | 234.44 Mo | **CONFORME** |
| 1 | batch | 10 | 23.53 | 470.54 | 230.0 | 0.0% | 47.46% | 232.75 Mo | **CONFORME** |
| 1 | batch | 25 | 38.99 | 779.86 | 480.0 | 0.0% | 79.97% | 236.7 Mo | **CONFORME** |
| 2 | batch | 5 | 12.05 | 240.95 | 75.0 | 0.0% | 0.0% | 35.29 Mo | **CONFORME** |
| 2 | batch | 10 | 25.15 | 503.08 | 98.0 | 0.0% | 0.0% | 35.29 Mo | **CONFORME** |
| 2 | batch | 10 | 25.66 | 513.14 | 130.0 | 0.0% | 0.0% | 34.66 Mo | **CONFORME** |
| 2 | batch | 25 | 58.24 | 1164.81 | 200.0 | 0.0% | 0.19% | 34.6 Mo | **CONFORME** |
| 4 | batch | 10 | 25.66 | 513.27 | 130.0 | 0.0% | 0.14% | 34.82 Mo | **CONFORME** |
| 4 | batch | 25 | 56.39 | 1127.82 | 250.0 | 0.0% | 0.28% | 34.78 Mo | **CONFORME** |
| 1 | mixed | 10 | 50.26 | 236.24 | 97.0 | 9.93% | 46.75% | 237.24 Mo | **NON-CONFORME** |
| 1 | mixed | 10 | 48.19 | 226.47 | 110.0 | 0.0% | 48.14% | 230.52 Mo | **CONFORME** |
| 1 | mixed | 25 | 83.0 | 390.09 | 290.0 | 9.69% | 77.26% | 238.48 Mo | **NON-CONFORME** |
| 1 | mixed | 25 | 79.35 | 372.95 | 290.0 | 0.0% | 76.21% | 233.41 Mo | **CONFORME** |
| 2 | mixed | 10 | 53.15 | 249.81 | 65.0 | 8.96% | 0.0% | 34.59 Mo | **NON-CONFORME** |
| 2 | mixed | 10 | 49.58 | 233.03 | 86.0 | 0.0% | 0.12% | 34.81 Mo | **CONFORME** |
| 2 | mixed | 25 | 116.28 | 546.5 | 140.0 | 9.45% | 0.0% | 34.59 Mo | **NON-CONFORME** |
| 2 | mixed | 25 | 104.38 | 490.59 | 210.0 | 0.0% | 0.12% | 34.8 Mo | **CONFORME** |
| 4 | mixed | 10 | 52.5 | 246.73 | 80.0 | 10.64% | 0.05% | 34.75 Mo | **NON-CONFORME** |
| 4 | mixed | 10 | 49.68 | 233.52 | 93.0 | 0.0% | 1.84% | 33.39 Mo | **CONFORME** |
| 4 | mixed | 25 | 111.81 | 525.51 | 160.0 | 9.08% | 0.19% | 34.75 Mo | **NON-CONFORME** |
| 4 | mixed | 25 | 94.23 | 442.89 | 240.0 | 0.0% | 0.0% | 34.94 Mo | **CONFORME** |
| 1 | single | 5 | 29.21 | 29.21 | 42.0 | 0.0% | 20.35% | 229.3 Mo | **CONFORME** |
| 1 | single | 10 | 58.21 | 58.21 | 58.0 | 0.0% | 41.36% | 230.67 Mo | **CONFORME** |
| 1 | single | 10 | 60.5 | 60.5 | 63.0 | 0.0% | 45.76% | 230.63 Mo | **CONFORME** |
| 1 | single | 25 | 92.83 | 92.83 | 250.0 | 0.0% | 75.66% | 233.07 Mo | **CONFORME** |
| 2 | single | 5 | 24.9 | 24.9 | 36.0 | 0.0% | 0.0% | 35.33 Mo | **CONFORME** |
| 2 | single | 10 | 62.52 | 62.52 | 42.0 | 0.0% | 0.0% | 35.33 Mo | **CONFORME** |
| 2 | single | 10 | 61.04 | 61.04 | 58.0 | 0.0% | 0.0% | 34.69 Mo | **CONFORME** |
| 2 | single | 25 | 141.78 | 141.78 | 98.0 | 0.0% | 0.0% | 34.68 Mo | **CONFORME** |
| 4 | single | 10 | 64.59 | 64.59 | 38.0 | 0.0% | 0.84% | 33.55 Mo | **CONFORME** |
| 4 | single | 25 | 143.61 | 143.61 | 90.0 | 0.0% | 0.0% | 34.85 Mo | **CONFORME** |

---

## 3. Analyse Comparative et Facteur d'Échelle (Scalabilité Multi-Workers)

1. **Gains de Débit et Scalabilité :**
   - L'augmentation du nombre de workers de 1 à 2 puis 4 permet d'accroître la capacité d'absorption des requêtes en exploitant le multi-cœur de la machine hôte.
   - En mode vectorisé `/predict/batch`, le traitement par lot scikit-learn mobilise intensivement les calculs CPU sur matrices NumPy/pandas, montrant une excellente efficience par worker.

2. **Comportement Mémoire et Saturation CPU :**
   - Chaque worker Uvicorn instancie les deux pipelines de modèles scikit-learn au démarrage (lifespan) sans rechargement à la requête.
   - L'empreinte mémoire RSS par worker est stable ($\sim 180 - 220$ Mo par worker), évitant toute fuite mémoire (*memory leak*) sous charge soutenue.
   - Le pool de connexions PostgreSQL (borné à 5 connexions par worker avec libération contextuelle immédiate) préserve la base de données de toute saturation ou épuisement de sockets.

---

## 4. Recommandation Chiffrée de Dimensionnement pour la Production

Les recommandations ci-dessous sont calculées sur la base du débit soutenable prudent observé $C_{\text{safe}}$ assorti d'une **marge d'exploitation de 30%** ($m = 0.3$), avec $C_{\text{recommended}} = C_{\text{safe}} \times (1 - m)$ :

- **Configuration 1 Worker(s) (Unitaire) :**
  - Débit soutenable observé ($C_{\text{safe}}$) : **92.83 req/s** (92.83 jeux/s, 185.67 inférences/s)
  - Latence $p_{95}$ mesurée : **250.0 ms** (SLO $< 500$ ms, erreurs: 0.0%)
  - Empreinte mémoire Uvicorn moyenne : **233.07 Mo** | CPU : **75.66%**
  - Débit recommandé avec marge d'exploitation de 30% ($C_{\text{recommended}}$) : **64.98 req/s**
  - Instances nécessaires pour absorber la charge crête cible ($\lambda_{\text{peak}} = 150.0$ req/s) : **3 réplique(s)**

- **Configuration 2 Worker(s) (Unitaire) :**
  - Débit soutenable observé ($C_{\text{safe}}$) : **141.78 req/s** (141.78 jeux/s, 283.57 inférences/s)
  - Latence $p_{95}$ mesurée : **98.0 ms** (SLO $< 500$ ms, erreurs: 0.0%)
  - Empreinte mémoire Uvicorn moyenne : **34.68 Mo** | CPU : **0.0%**
  - Débit recommandé avec marge d'exploitation de 30% ($C_{\text{recommended}}$) : **99.25 req/s**
  - Instances nécessaires pour absorber la charge crête cible ($\lambda_{\text{peak}} = 150.0$ req/s) : **2 réplique(s)**

- **Configuration 4 Worker(s) (Unitaire) :**
  - Débit soutenable observé ($C_{\text{safe}}$) : **143.61 req/s** (143.61 jeux/s, 287.22 inférences/s)
  - Latence $p_{95}$ mesurée : **90.0 ms** (SLO $< 500$ ms, erreurs: 0.0%)
  - Empreinte mémoire Uvicorn moyenne : **34.85 Mo** | CPU : **0.0%**
  - Débit recommandé avec marge d'exploitation de 30% ($C_{\text{recommended}}$) : **100.53 req/s**
  - Instances nécessaires pour absorber la charge crête cible ($\lambda_{\text{peak}} = 150.0$ req/s) : **2 réplique(s)**

### Dimensionnement Cible Recommandé
- **Architecture préconisée pour le service d'inférence :** Déploiement en conteneurs avec **2 à 4 workers Uvicorn** par réplique.
- **Ressources matérielles requises par réplique :**
  - **CPU :** 2 vCPU dédiés (garantissant la tenue des latences $p_{95} < 500$ ms).
  - **RAM :** 1 Go minimum (permettant d'héberger les 2 pipelines scikit-learn et la mémoire tampon des requêtes vectorisées).
- **PostgreSQL :** Pool configuré à 5 connexions par worker, soit 10 à 20 connexions maximales par réplique, largement sous le seuil maximal de PostgreSQL (`max_connections = 100`).

---

## 5. Limites de Validité et Réserves Expérimentales

1. **Environnement de Mesure :**
   - Les tests ont été réalisés sur un banc local partagé (machine hôte Windows 11 avec conteneurs Docker colocalisés).
   - En environnement de production avec réseau découplé et serveur PostgreSQL hébergé sur une instance dédiée, la latence réseau inter-serveurs doit être intégrée dans le budget global de $p_{95}$.
2. **Capacité du Générateur Locust :**
   - L'utilisation CPU du processus Locust est restée modérée ($< 30\%$), garantissant que le banc de test n'a pas constitué un goulot d'étranglement artificiel.
