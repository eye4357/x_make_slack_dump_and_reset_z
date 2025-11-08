# Security & Secret Handling

This tool (`x_make_slack_dump_and_reset_z`) permanently isolates the persistent environment variable helper.

Hard guarantees:
- The persistent helper (`x_make_persistent_env_var_x`) is NEVER imported at runtime here.
- There is NO environment variable, flag, or CLI option to enable it.
- Secret sourcing order is strictly:
  1. `parameters.slack_token` (required unless env provides one)
  2. `SLACK_TOKEN` environment variable
- Persistent vault lookup is inert (factory is always `None`).

Rationale: eliminate accidental credential persistence, reduce attack surface, and satisfy least‑privilege / CISSP constraints. Use the helper only via a standalone, manual invocation in a controlled context if absolutely required.
