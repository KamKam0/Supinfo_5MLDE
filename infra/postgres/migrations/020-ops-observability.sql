-- 020-ops-observability.sql : Migration du schéma d'observabilité OPS
-- Projet MLOps Steam 5MLDE

BEGIN;

-- 1. Création / Mise à jour du rôle technique ops_writer
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'ops_writer') THEN
        CREATE ROLE ops_writer WITH LOGIN PASSWORD '5mlde-local-ops';
    ELSE
        ALTER ROLE ops_writer WITH PASSWORD '5mlde-local-ops';
    END IF;
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'grafana_reader') THEN
        CREATE ROLE grafana_reader WITH LOGIN PASSWORD '5mlde-local-grafana';
    END IF;
END
$$;

-- 2. Création du schéma ops
CREATE SCHEMA IF NOT EXISTS ops;

-- Permissions de connexion à la base et usage du schéma
GRANT CONNECT ON DATABASE steam_api TO ops_writer;
REVOKE ALL ON SCHEMA ops FROM PUBLIC;
GRANT USAGE ON SCHEMA ops TO ops_writer, grafana_reader;

-- 3. Table des exécutions de pipeline (ops.pipeline_runs)
CREATE TABLE IF NOT EXISTS ops.pipeline_runs (
    run_id uuid PRIMARY KEY,
    flow_run_id text NOT NULL UNIQUE,
    flow_name text NOT NULL,
    batch_id text NOT NULL,
    mlflow_run_id text,
    publication_id text,
    started_at timestamptz NOT NULL,
    completed_at timestamptz,
    duration_seconds double precision
        CHECK (duration_seconds IS NULL OR (duration_seconds >= 0 AND duration_seconds < 'Infinity'::double precision)),
    trigger_mode text NOT NULL
        CHECK (trigger_mode IN ('manual', 'push_threshold', 'scheduled', 'api_event', 'test')),
    input_snapshot text,
    raw_records_count integer CHECK (raw_records_count IS NULL OR raw_records_count >= 0),
    eligible_records_count integer CHECK (eligible_records_count IS NULL OR eligible_records_count >= 0),
    terminal_state text NOT NULL
        CHECK (terminal_state IN ('COMPLETED', 'FAILED', 'CANCELLED', 'RUNNING')),
    gx_validation_passed boolean,
    gx_offending_appids jsonb,
    error_message text,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);

COMMENT ON TABLE ops.pipeline_runs IS
    'Suivi des exécutions du pipeline d entraînement Prefect, corrélation universelle et audit de conformité GX.';

-- 4. Table des tâches d'orchestration (ops.pipeline_tasks)
CREATE TABLE IF NOT EXISTS ops.pipeline_tasks (
    task_run_id text PRIMARY KEY,
    flow_run_id text NOT NULL REFERENCES ops.pipeline_runs(flow_run_id) ON DELETE CASCADE,
    task_name text NOT NULL,
    started_at timestamptz NOT NULL,
    completed_at timestamptz,
    duration_seconds double precision
        CHECK (duration_seconds IS NULL OR (duration_seconds >= 0 AND duration_seconds < 'Infinity'::double precision)),
    terminal_state text NOT NULL
        CHECK (terminal_state IN ('COMPLETED', 'FAILED', 'CANCELLED', 'RUNNING')),
    error_message text,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);

COMMENT ON TABLE ops.pipeline_tasks IS
    'Détail unitaire des 4 tâches mères Prefect (load, preprocess, train, publish).';

-- 5. Table de suivi de l'exporteur et fraîcheur de télémétrie (ops.export_sync_state)
CREATE TABLE IF NOT EXISTS ops.export_sync_state (
    exporter_id text PRIMARY KEY,
    last_sync_at timestamptz NOT NULL,
    last_scanned_event_at timestamptz,
    records_exported integer NOT NULL DEFAULT 0,
    status text NOT NULL DEFAULT 'OK',
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp()
);

COMMENT ON TABLE ops.export_sync_state IS
    'Horodatage de la dernière synchronisation réussie pour mesurer le décalage temporel OPS.';

-- 6. Index d'observabilité pour accélérer les requêtes Grafana sans contention
CREATE INDEX IF NOT EXISTS idx_ops_pipeline_runs_started_at
    ON ops.pipeline_runs (started_at DESC);

CREATE INDEX IF NOT EXISTS idx_ops_pipeline_runs_terminal_state
    ON ops.pipeline_runs (terminal_state);

CREATE INDEX IF NOT EXISTS idx_ops_pipeline_runs_trigger_mode
    ON ops.pipeline_runs (trigger_mode);

CREATE INDEX IF NOT EXISTS idx_ops_pipeline_tasks_flow_run_id
    ON ops.pipeline_tasks (flow_run_id);

CREATE INDEX IF NOT EXISTS idx_ops_pipeline_tasks_task_name
    ON ops.pipeline_tasks (task_name);

-- 7. Attribution stricte des privilèges (principe de moindre privilège)
-- ops_writer : lecture, insertion et mise à jour dans ops
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA ops TO ops_writer;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA ops TO ops_writer;

-- grafana_reader : lecture seule avec interdiction absolue d'écriture
GRANT SELECT ON ALL TABLES IN SCHEMA ops TO grafana_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA ops GRANT SELECT ON TABLES TO grafana_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA ops GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO ops_writer;

COMMIT;
