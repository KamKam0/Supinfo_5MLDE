-- infra/grafana/queries/api_durations.sql
-- Percentiles de latence et temps d'inférence pure
-- Règle d'or : percentile_cont SQL exact, sans multiplication cartésienne

-- 1. Percentiles de traitement HTTP complet (processing_duration_ms)
SELECT
    endpoint,
    count(*) AS sample_size,
    round(avg(processing_duration_ms)::numeric, 2) AS mean_ms,
    round(percentile_cont(0.50) WITHIN GROUP (ORDER BY processing_duration_ms)::numeric, 2) AS p50_ms,
    round(percentile_cont(0.95) WITHIN GROUP (ORDER BY processing_duration_ms)::numeric, 2) AS p95_ms,
    round(percentile_cont(0.99) WITHIN GROUP (ORDER BY processing_duration_ms)::numeric, 2) AS p99_ms,
    round(max(processing_duration_ms)::numeric, 2) AS max_ms
FROM tracing.api_requests
GROUP BY endpoint
ORDER BY endpoint;

-- 2. Percentiles d'inférence pure (inference_duration_ms) sur les prédictions réussies
SELECT
    endpoint,
    count(*) AS sample_size,
    round(avg(inference_duration_ms)::numeric, 2) AS mean_inference_ms,
    round(percentile_cont(0.50) WITHIN GROUP (ORDER BY inference_duration_ms)::numeric, 2) AS p50_inference_ms,
    round(percentile_cont(0.95) WITHIN GROUP (ORDER BY inference_duration_ms)::numeric, 2) AS p95_inference_ms,
    round(percentile_cont(0.99) WITHIN GROUP (ORDER BY inference_duration_ms)::numeric, 2) AS p99_inference_ms
FROM tracing.api_requests
WHERE inference_duration_ms IS NOT NULL
  AND http_status = 200
GROUP BY endpoint
ORDER BY endpoint;

-- 3. Ventilation de la latence par taille de lot (item_count)
SELECT
    item_count,
    count(*) AS batch_requests_count,
    round(avg(processing_duration_ms)::numeric, 2) AS avg_duration_ms,
    round(percentile_cont(0.95) WITHIN GROUP (ORDER BY processing_duration_ms)::numeric, 2) AS p95_duration_ms
FROM tracing.api_requests
WHERE endpoint = '/predict/batch'
  AND http_status = 200
GROUP BY item_count
ORDER BY item_count;
