# CSMAR Root Data Source

This directory is the canonical root-level CSMAR code path for PIT financial data, announcement dates, and field-level financial factor rebuild preparation.

Credentials are loaded from environment variables first, then root `.env.local`, then legacy `xhs/.env.local`. Credential values must not be printed, logged, or committed.

Legacy CSMAR scripts and historical outputs under `xhs/` remain historical references.

