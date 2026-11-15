# Rapport d'Incident : Dégradation de Modèle & Rollback d'Urgence (Incident E)

- **Date :** 2026-11-16T17:50:11.866839+00:00
- **Scénario :** Déploiement d'une version régressant en précision et retour arrière sans interruption destructrice
- **Composant testé :** Manifeste de service immuable (`state/service-manifest.json`), scripts de sélection et traçabilité SQL
- **Statut final :** ✅ SUCCÈS (Rétablissement vérifié du modèle A, conservation intégrale de l'historique d'audit)

---

## 1. Chronologie de l'Incident
1. **État Nominal (Phase A) :**
   - Manifeste actif : `man-a7f270fb` (Régression: `v12`, Classification: `v12`)
   - Requête témoin : `f38ad4b7-2b35-4227-933d-087159162dff` servie avec succès.
2. **Déploiement Incident (Phase B) :**
   - Déploiement du manifeste `man-degraded-77b070` (simulation de version dégradée).
   - Requête servie : `7fe1e9cb-265a-4fc1-9248-03b1d9be0d0a` portant l'empreinte du modèle B.
3. **Activation de la Procédure de Retour Arrière (Rollback) :**
   - Restauration atomique de `state/service-manifest.prev.json` vers `state/service-manifest.json`.
   - Réalignement des alias `@champion` dans le registre de modèles MLflow.
   - Rechargement de l'API.
4. **Vérification Post-Rollback :**
   - Requête d'inférence `319b3565-d1ed-4ed1-8e7f-274d6bd5bf4b` servie par le manifeste réactivé `man-a7f270fb`.

## 2. Règle d'Or de Traçabilité & Non-Suppression d'Historique
L'inspection de la table `tracing.api_requests` confirme :
- Les prédictions émises sous le Manifeste A restent associées aux identifiants du modèle A.
- Les prédictions émises sous le Manifeste B **ne sont pas écrasées** et conservent la trace du modèle B pour l'analyse post-mortem.
- Les nouvelles prédictions post-rollback tracent à nouveau le modèle A.
- Total des requêtes d'audit réconciliées en base : **3 / 3**.
