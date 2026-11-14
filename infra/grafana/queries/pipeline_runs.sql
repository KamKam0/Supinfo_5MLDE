-- infra/grafana/queries/pipeline_runs.sql
-- Métriques des entraînements Prefect, conformité Great Expectations et fraîcheur OPS
-- Règle d'or : Runs distincts, dernier état par run, décalage temporel explicite

-- 1. Vue synthétique des entraînements : total, succès, échecs et durée moyenne
SELECT
    count(*) AS total_pipeline_runs,
    count(*) FILTER (WHERE terminal_state = 'COMPLETED') AS successful_runs,
    count(*) FILTER (WHERE terminal_state = 'FAILED') AS failed_runs,
    count(*) FILTER (WHERE terminal_state = 'RUNNING') AS active_runs,
    round(
        (100.0 * count(*) FILTER (WHERE terminal_state = 'COMPLETED') / NULLIF(count(*), 0))::numeric,
        2
    ) AS success_rate_pct,
    round(avg(duration_seconds) FILTER (WHERE terminal_state = 'COMPLETED')::numeric, 1) AS avg_duration_completed_sec
FROM ops.pipeline_runs;

-- 2. Audit de conformité Great Expectations (rejets et blocages amont)
SELECT
    count(*) AS total_validations_audited,
    count(*) FILTER (WHERE gx_validation_passed = true) AS gx_passed_count,
    count(*) FILTER (WHERE gx_validation_passed = false) AS gx_rejected_count,
    round(
        (100.0 * count(*) FILTER (WHERE gx_validation_passed = true) / NULLIF(count(*), 0))::numeric,
        2
    ) AS gx_compliance_rate_pct
FROM ops.pipeline_runs;

-- 3. Ventilation par mode de déclenchement (trigger_mode)
SELECT
    trigger_mode,
    count(*) AS runs_count,
    count(*) FILTER (WHERE terminal_state = 'COMPLETED') AS completed_count,
    count(*) FILTER (WHERE terminal_state = 'FAILED') AS failed_count,
    round(avg(duration_seconds)::numeric, 1) AS avg_duration_sec
FROM ops.pipeline_runs
GROUP BY trigger_mode
ORDER BY runs_count DESC;

-- 4. Fraîcheur d'export OPS et décalage temporel
SELECT
    exporter_id,
    last_sync_at,
    records_exported,
    status,
    round(EXTRACT(EPOCH FROM (now() - last_sync_at))::numeric, 1) AS sync_lag_seconds
FROM ops.export_sync_state;

-- 5. Durée d'exécution détaillée par tâche Prefect
SELECT
    task_name,
    count(*) AS execution_count,
    round(avg(duration_seconds)::numeric, 2) AS avg_duration_sec,
    round(percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_seconds)::numeric, 2) AS p95_duration_sec
FROM ops.pipeline_tasks
WHERE duration_seconds IS NOT NULL
GROUP BY task_name
ORDER BY avg_duration_sec DESC;
