# Rapport d'Incident : Crash et Reprise de PostgreSQL (Incident B)

- **Date :** 2026-11-16T17:47:05.082162+00:00
- **Scénario :** Indisponibilité inopinée du service PostgreSQL (Port 5432)
- **Composant testé :** API FastAPI d'Inférence, sondes de santé et télémétrie de secours
- **Statut final :** ✅ SUCCÈS (Mode dégradé 503 sans faux 200, repli local vérifié, reconnexion automatique)

---

## 1. Contexte Initial
- Sonde de vivacité `/health/live` : HTTP 200
- Sonde d'aptitude `/health/ready` : HTTP 200
- Requête d'inférence nominale `/predict` : HTTP 200 (Request ID: `2e922068-e840-4b5b-b9b5-c1fcfc6cbc21`)

## 2. Injection de Panne
Arrêt du service `postgres` (`docker compose stop postgres`).

## 3. Comportement Observé Pendant la Panne
- **Vivacité (`/health/live`) :** HTTP 200 (le processus reste actif pour répondre aux sondes d'infrastructure).
- **Aptitude (`/health/ready`) :** HTTP 503 (`DEPENDENCY_UNAVAILABLE` : base inaccessible).
- **Inférence métier (`/predict`) :** HTTP 503
  - **Preuve anti-faux-succès :** Aucun code HTTP 200 n'a été retourné au client.
  - **Télémétrie locale :** L'incident d'inférence a été consigné dans `state/telemetry_fallback.jsonl`.

## 4. Procédure de Rétablissement
1. Redémarrage du conteneur PostgreSQL (`docker compose start postgres`).
2. Attente de la réouverture des sockets TCP et du healthcheck SQL (`SELECT 1`).
3. Reconnexion automatique du pool de connexions (`DatabasePool`) sans nécessiter de redémarrage des workers Uvicorn.

## 5. Vérification Post-Incident
- **Aptitude (`/health/ready`) :** HTTP 200 (Prêt)
- **Inférence après reprise (`/predict`) :** HTTP 200 (Request ID: `3170bb7b-1514-403f-879e-34794a655670`)
- **Intégrité transactionnelle :** Transaction confirmée et persistée dans PostgreSQL `tracing.api_requests`.
