# x_make_slack_dump_and_reset_z — Slack Purge Lab Manual

Listen closely: this playbook moves every syllable and pixel from Slack into the Change Control vault, then wipes the channels so the next sprint starts clean. No drift, no ghosts, no excuses.

## Mission Objectives
- Drain configured Slack channels into the freshest Change Control folder.
- Preserve message context, threads, and attachments as JSON artifacts.
- Optionally vaporize the source messages and files once archived.
- Emit machine-verifiable ledgers so the orchestrator can certify the burn.

## Field Kit
- Python 3.11+
- `requests` and the shared utilities from `x_make_common_x`
- Slack Web API token with scopes capable of reading history, downloading files, and deleting evidence

## Deployment Steps
1. `python -m venv .venv`
2. `\.venv\Scripts\Activate.ps1`
3. `pip install -r requirements.txt`
4. `python -m x_make_slack_dump_and_reset_z`

Feed the runner a JSON payload describing channel IDs, archive root, and how aggressive the purge should be. The export lands inside the newest Change Control folder, under a timestamped subdirectory.

## Verification Grid
| Check | Command |
| --- | --- |
| Formatter | `python -m black .` |
| Lint | `python -m ruff check .` |
| Types | `python -m mypy .` |
| Static Contracts | `python -m pyright` |
| Tests | `pytest`

## Compliance Notes
- Only deploy tokens owned by workspace administrators—Slack refuses to let tourists delete other people’s data.
- Respect retention obligations. If the law demands a trail, toggle `delete_after_export` to `false` and let retention handle the rest.
- Archive volumes add up; schedule offload from Change Control to cheaper storage if the vault starts bulking up.

## Provenance
Built in the lab. The Heisenberg tone stays because clarity cuts through the noise.
