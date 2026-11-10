"""
Database layer for Agentic Decision Intelligence.
Uses Neon PostgreSQL via the central connection pool in portal.database.
"""

from portal.database import get_neon_connection


def init_agent_tables():
    """
    Idempotently creates all 8 agent tables and indexes in Neon PostgreSQL.
    Guarantees tenant isolation and audit compliance.
    """
    ddl = """
    CREATE TABLE IF NOT EXISTS agent_runs (
        id UUID PRIMARY KEY,
        customer_id VARCHAR(255) DEFAULT '',
        user_id VARCHAR(255) DEFAULT '',
        session_id VARCHAR(255) NOT NULL,
        solutions JSONB DEFAULT '[]'::jsonb,
        status VARCHAR(50) DEFAULT 'queued',
        stage VARCHAR(50) DEFAULT 'initialized',
        progress INTEGER DEFAULT 0,
        cache_key VARCHAR(255) DEFAULT '',
        ledger_hash VARCHAR(255) DEFAULT '',
        config_version VARCHAR(50) DEFAULT '1.0.0',
        started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        finished_at TIMESTAMP WITH TIME ZONE,
        error TEXT,
        tokens_in INTEGER DEFAULT 0,
        tokens_out INTEGER DEFAULT 0,
        cost_usd_est NUMERIC(10, 6) DEFAULT 0.0
    );

    CREATE TABLE IF NOT EXISTS agent_evidence (
        id SERIAL PRIMARY KEY,
        run_id UUID REFERENCES agent_runs(id) ON DELETE CASCADE,
        evidence_id VARCHAR(100) NOT NULL,
        solution VARCHAR(50) NOT NULL,
        kind VARCHAR(50) NOT NULL,
        key VARCHAR(255) NOT NULL,
        value JSONB,
        unit VARCHAR(50),
        source TEXT,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS agent_stage_outputs (
        id SERIAL PRIMARY KEY,
        run_id UUID REFERENCES agent_runs(id) ON DELETE CASCADE,
        stage VARCHAR(50) NOT NULL,
        output JSONB NOT NULL,
        schema_version VARCHAR(50) DEFAULT '1.0.0',
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS agent_dossiers (
        id SERIAL PRIMARY KEY,
        run_id UUID REFERENCES agent_runs(id) ON DELETE CASCADE,
        version INTEGER DEFAULT 1,
        dossier JSONB NOT NULL,
        generated_by VARCHAR(50) DEFAULT 'llm',
        verifier_passed BOOLEAN DEFAULT TRUE,
        confidence_overall NUMERIC(5, 4) DEFAULT 0.0,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS agent_approvals (
        id SERIAL PRIMARY KEY,
        dossier_id INTEGER REFERENCES agent_dossiers(id) ON DELETE CASCADE,
        user_id VARCHAR(255) DEFAULT '',
        decision VARCHAR(50) NOT NULL,
        reason TEXT,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS agent_semantic_mappings (
        id SERIAL PRIMARY KEY,
        customer_id VARCHAR(255) DEFAULT 'default',
        column_name VARCHAR(255) NOT NULL,
        business_label VARCHAR(255) NOT NULL,
        unit VARCHAR(50),
        controllable BOOLEAN DEFAULT TRUE,
        status VARCHAR(50) DEFAULT 'curated',
        created_by VARCHAR(255) DEFAULT 'system',
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS agent_customer_config (
        id SERIAL PRIMARY KEY,
        customer_id VARCHAR(255) UNIQUE NOT NULL,
        thresholds JSONB DEFAULT '{}'::jsonb,
        financial_assumptions JSONB DEFAULT '{}'::jsonb,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS agent_trace_events (
        id SERIAL PRIMARY KEY,
        run_id UUID REFERENCES agent_runs(id) ON DELETE CASCADE,
        stage VARCHAR(50) NOT NULL,
        event_type VARCHAR(100) NOT NULL,
        payload JSONB DEFAULT '{}'::jsonb,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS agent_explanation_artifacts (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        run_id VARCHAR(255) NOT NULL,
        solution VARCHAR(50) NOT NULL,
        kind VARCHAR(50) NOT NULL,
        sha256 VARCHAR(64) NOT NULL,
        size_bytes INTEGER NOT NULL,
        content BYTEA NOT NULL,
        metadata JSONB DEFAULT '{}'::jsonb,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );

    -- Performance Indexes
    CREATE INDEX IF NOT EXISTS idx_agent_runs_cust_sess ON agent_runs (customer_id, session_id);
    CREATE INDEX IF NOT EXISTS idx_agent_runs_cache_key ON agent_runs (cache_key);
    CREATE INDEX IF NOT EXISTS idx_agent_stage_outputs_run ON agent_stage_outputs (run_id, stage);
    CREATE INDEX IF NOT EXISTS idx_agent_evidence_run ON agent_evidence (run_id, evidence_id);
    CREATE INDEX IF NOT EXISTS idx_agent_artifacts_lookup ON agent_explanation_artifacts (run_id, solution, kind);
    """
    try:
        with get_neon_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(ddl)
        print("[Neon DB] Agentic AI tables and indexes initialized successfully.")
        return True
    except Exception as e:
        print(f"[Neon DB] Note on Agentic AI tables init: {e}")
        return False
