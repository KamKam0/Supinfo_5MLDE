-- infra/grafana/queries/api_errors.sql
-- Surveillance des taux d'erreurs HTTP 4xx et 5xx
-- Règle d'or : Dénominateur exact basé sur le total des tentatives HTTP

-- 1. Taux d'erreurs global et par endpoint
SELECT
    endpoint,
    count(*) AS total_attempts,
    count(*) FILTER (WHERE http_status BETWEEN 200 AND 299) AS count_2xx,
    count(*) FILTER (WHERE http_status BETWEEN 400 AND 499) AS count_4xx,
    count(*) FILTER (WHERE http_status >= 500) AS count_5xx,
    round(
        (100.0 * count(*) FILTER (WHERE http_status BETWEEN 400 AND 499) / NULLIF(count(*), 0))::numeric,
        2
    ) AS error_rate_4xx_pct,
    round(
        (100.0 * count(*) FILTER (WHERE http_status >= 500) / NULLIF(count(*), 0))::numeric,
        2
    ) AS error_rate_5xx_pct
FROM tracing.api_requests
GROUP BY endpoint
ORDER BY endpoint;

-- 2. Répartition par code d'erreur applicatif
SELECT
    COALESCE(error_code, 'NONE_SUCCESS') AS error_code,
    http_status,
    endpoint,
    count(*) AS occurrences,
    max(started_at) AS last_occurrence_at
FROM tracing.api_requests
WHERE http_status >= 400
GROUP BY error_code, http_status, endpoint
ORDER BY occurrences DESC;
