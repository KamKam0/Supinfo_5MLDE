-- infra/grafana/queries/api_usage.sql
-- Métriques de trafic, débit et volumétrie d'inférence
-- Règles d'or : aucune jointure multiplicative avec predictions, somme de item_count sur requêtes 200 uniquement

-- 1. Volume total de requêtes et tentatives par endpoint
SELECT
    endpoint,
    count(*) AS total_requests,
    count(*) FILTER (WHERE http_status = 200) AS successful_requests,
    count(*) FILTER (WHERE http_status <> 200) AS failed_requests,
    COALESCE(sum(item_count) FILTER (WHERE http_status = 200 AND endpoint IN ('/predict', '/predict/batch')), 0) AS total_games_predicted
FROM tracing.api_requests
GROUP BY endpoint
ORDER BY endpoint;

-- 2. Débit moyen par seconde et ventilation par campagne
SELECT
    campaign_id,
    count(*) AS requests_count,
    COALESCE(sum(item_count) FILTER (WHERE http_status = 200 AND endpoint IN ('/predict', '/predict/batch')), 0) AS games_count,
    min(started_at) AS first_request_at,
    max(started_at) AS last_request_at,
    round(
        count(*)::numeric / NULLIF(EXTRACT(EPOCH FROM (max(started_at) - min(started_at))), 0)::numeric,
        2
    ) AS req_per_second
FROM tracing.api_requests
GROUP BY campaign_id
ORDER BY requests_count DESC;

-- 3. Répartition de la charge par worker (instance_id, worker_pid)
SELECT
    instance_id,
    worker_pid,
    count(*) AS requests_handled,
    min(started_at) AS first_seen,
    max(started_at) AS last_seen
FROM tracing.api_requests
GROUP BY instance_id, worker_pid
ORDER BY requests_handled DESC;
