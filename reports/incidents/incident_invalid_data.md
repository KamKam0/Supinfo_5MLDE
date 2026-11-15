# Rapport d'Incident : Injection de Données Invalides (Incident A)

- **Date :** 2026-11-16T17:42:24.218456+00:00
- **Scénario :** Ingestion d'un lot corrompu (prix textuel, appid dupliqué, pourcentage hors bornes)
- **Composant testé :** Barrière de données Great Expectations Core 1.x (`src/data_validation.py`)
- **Statut final :** ✅ SUCCÈS (Incident intercepté et contenu sans corruption du système)

---

## 1. Contexte & Préconditions
- **Empreinte du manifeste actif avant injection :** `16c282530ef2aa17be63c8d1bf852fd23b2f682f20f6ef06f1d821e255fb4e35`
- **Politique de validation :** Suite `historical_schema_v1` avec levée bloquante `DataValidationError`.

## 2. Injection Réalisée
Deux enregistrements corrompus ont été soumis à la barrière de validation :
1. `Corrupted Game Alpha` : `price = "Gratuit"` (type string interdit).
2. `Corrupted Game Beta` : `appid = 999001` (duplicata strict) et `pct_pos_total = 150.0` (dépassement de la borne [0, 100]).

## 3. Observations & Détections
- **Interception Great Expectations :** OUI (bloquant)
- **Exception levée :** `DataValidationError`
- **Extrait de l'erreur :**
```text
[QUALITÉ DONNÉES] Rejet bloquant par Great Expectations (historical_schema_v1) : 5 contrôle(s) non satisfait(s). Rapport archivé à : validation_failure_historical_schema_v1_2026-11-16T17-42-21_994673+00-00.json
```
- **Alerte transmise :** Notification émise via `trigger_failure_alert`.
- **Exécution aval :** Aucune étape de prétraitement, d'entraînement ou de sélection n'a été déclenchée.

## 4. Vérification Post-Incident
- **Empreinte du manifeste actif après injection :** `16c282530ef2aa17be63c8d1bf852fd23b2f682f20f6ef06f1d821e255fb4e35`
- **Conservation stricte des modèles :** Modèles inchangés
- **Conclusion :** La barrière bloquante a protégé l'intégrité de la chaîne d'entraînement sans polluer le registre de modèles.
