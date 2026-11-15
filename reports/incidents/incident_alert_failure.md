# Rapport d'Incident : Indisponibilité du Récepteur d'Alertes (Incident F)

- **Date :** 2026-11-16T17:50:27.228962+00:00
- **Scénario :** Panne de destination / webhook d'alerte inaccessible (`http://127.0.0.1:59999/unreachable_alert_webhook`)
- **Composant testé :** Dispatcheur d'alertes résilient (`src/orchestration/alerts.py`)
- **Statut final :** ✅ SUCCÈS (Erreur interceptée, temps borné, absence de boucle de retry destructive)

---

## 1. Contexte & Défaillance
Lorsqu'un échec de pipeline survient, le système tente de notifier le canal d'astreinte.
Dans ce scénario, la destination configurée est volontairement indisponible (port fermé).

## 2. Comportement Observé du Dispatcheur
- **Tentative d'envoi :** Déclenchée avec timeout strict (1.0 s).
- **Statut retourné :** `status = CONNECTION_FAILED`.
- **Confinement d'erreur :** L'exception réseau a été capturée et enregistrée dans les logs d'exploitation sans être propagée au pipeline parent.

## 3. Règle d'Or MLOps Respectée
L'échec du système de notification :
1. **N'a pas provoqué** le crash du worker Prefect ou du script d'entraînement.
2. **N'a pas déclenché** de relance aveugle du pipeline d'entraînement (ce qui aurait pu saturer le cluster de calcul).
3. Conserve l'état terminal `FAILED` original pour l'inspection humaine.
