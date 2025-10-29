from __future__ import annotations

from x_make_common_x.json_contracts import validate_payload

from x_make_slack_dump_and_reset_z.json_contracts import (
    ERROR_SCHEMA,
    INPUT_SCHEMA,
    OUTPUT_SCHEMA,
)


def test_input_schema_accepts_minimal_payload() -> None:
    payload = {
        "command": "x_make_slack_dump_and_reset_x",
        "parameters": {
            "channels": ["C123"],
            "archive_root": "./tmp",
        },
    }
    validate_payload(payload, INPUT_SCHEMA)


def test_output_schema_accepts_minimal_payload() -> None:
    payload = {
        "status": "success",
        "schema_version": "x_make_slack_dump_and_reset_x.run/1.0",
        "export_root": "./tmp/sprint",
        "channels": [],
    }
    validate_payload(payload, OUTPUT_SCHEMA)


def test_error_schema_accepts_message() -> None:
    payload = {
        "status": "failure",
        "message": "boom",
    }
    validate_payload(payload, ERROR_SCHEMA)
