# Rapport d'Incident : Publication Partielle MLflow & Réconciliation (Incident C)

- **Date :** 2026-11-16T17:47:55.621808+00:00
- **Scénario :** Rupture de connexion réseau en cours de publication Model Registry
- **Composant testé :** Moteur de réconciliation idempotente (`src/tracking/reconciliation.py`, `scripts/reconcile.py`)
- **Statut final :** ✅ SUCCÈS (Reprise déterministe, aucune version orpheline ni duplication)

---

## 1. Contexte & Défaillance Simulée
Lors de la double publication requise par le catalogue Steam (régression du pourcentage d'avis et classification de tranche de propriétaires) :
- Le modèle de régression `steam-positive-ratio` est enregistré (version `v12`) et reçoit l'alias `@test-reconcile-6400`.
- Un arrêt brutal survient avant l'enregistrement du modèle de classification `steam-owner-range`.
- État dans le journal SQLite (`data/sqlite/publications.db`) : `IN_PROGRESS`, `classification_version = None`.

## 2. Procédure de Réconciliation Exécutée
Commande opérateur :
```sh
uv run --locked python scripts/reconcile.py --publication-id pub-reconcile-inc-069219e3
```

## 3. Observations du Moteur
- **Détection des jalons existants :** Le moteur constate que la version `v12` est déjà présente dans MLflow et aliée ; il s'abstient rigoureusement de la recréer.
- **Achèvement du jalon manquant :** Création de la version de classification `v12` et affectation de l'alias `@test-reconcile-6400`.
- **Transition d'état :** Statut final promu à `COMPLETED`.

## 4. Preuve d'Idempotence Absolue
Une seconde exécution consécutive de réconciliation sur `pub-reconcile-inc-069219e3` confirme immédiatement :
`already_completed: True`, avec zéro version créée et zéro modification d'alias.
