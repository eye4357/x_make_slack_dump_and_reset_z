"""JSON contracts for x_make_slack_dump_and_reset_z."""

from __future__ import annotations

import sys as _sys

_JSON_VALUE_SCHEMA: dict[str, object] = {
    "type": ["object", "array", "string", "number", "boolean", "null"],
}

_NON_EMPTY_STRING: dict[str, object] = {"type": "string", "minLength": 1}

_CHANNEL_ITEM_SCHEMA: dict[str, object] = {
    "oneOf": [
        _NON_EMPTY_STRING,
        {
            "type": "object",
            "properties": {
                "id": _NON_EMPTY_STRING,
                "name": _NON_EMPTY_STRING,
            },
            "additionalProperties": False,
            "minProperties": 1,
        },
    ]
}

_PARAMETERS_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "slack_token": _NON_EMPTY_STRING,
        "channels": {
            "type": "array",
            "items": _CHANNEL_ITEM_SCHEMA,
            "minItems": 1,
        },
        "archive_root": _NON_EMPTY_STRING,
        "delete_after_export": {"type": "boolean"},
        "include_files": {"type": "boolean"},
        "include_threads": {"type": "boolean"},
        "dry_run": {"type": "boolean"},
        "skip_channels": {
            "type": "array",
            "items": _NON_EMPTY_STRING,
        },
        "notes": {
            "type": "array",
            "items": _NON_EMPTY_STRING,
        },
    },
    "required": ["channels", "archive_root"],
    "additionalProperties": False,
}

INPUT_SCHEMA: dict[str, object] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "x_make_slack_dump_and_reset_z input",
    "type": "object",
    "properties": {
        "command": {"const": "x_make_slack_dump_and_reset_x"},
        "parameters": _PARAMETERS_SCHEMA,
    },
    "required": ["command", "parameters"],
    "additionalProperties": False,
}

_CHANNEL_RESULT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "channel_id": _NON_EMPTY_STRING,
        "channel_name": _NON_EMPTY_STRING,
        "message_count": {"type": "integer", "minimum": 0},
        "file_count": {"type": "integer", "minimum": 0},
        "export_path": _NON_EMPTY_STRING,
        "deleted": {"type": "boolean"},
        "notes": {
            "type": "array",
            "items": _NON_EMPTY_STRING,
        },
    },
    "required": [
        "channel_id",
        "channel_name",
        "message_count",
        "file_count",
        "export_path",
        "deleted",
    ],
    "additionalProperties": False,
}

OUTPUT_SCHEMA: dict[str, object] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "x_make_slack_dump_and_reset_z output",
    "type": "object",
    "properties": {
        "status": {"const": "success"},
        "schema_version": {"const": "x_make_slack_dump_and_reset_x.run/1.0"},
        "export_root": _NON_EMPTY_STRING,
        "channels": {
            "type": "array",
            "items": _CHANNEL_RESULT_SCHEMA,
        },
        "messages": {
            "type": "array",
            "items": _NON_EMPTY_STRING,
        },
    },
    "required": ["status", "schema_version", "export_root", "channels"],
    "additionalProperties": False,
}

ERROR_SCHEMA: dict[str, object] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "x_make_slack_dump_and_reset_z error",
    "type": "object",
    "properties": {
        "status": {"const": "failure"},
        "message": _NON_EMPTY_STRING,
        "details": {
            "type": "object",
            "additionalProperties": _JSON_VALUE_SCHEMA,
        },
    },
    "required": ["status", "message"],
    "additionalProperties": True,
}

_sys.modules.setdefault("json_contracts", _sys.modules[__name__])

__all__ = ["ERROR_SCHEMA", "INPUT_SCHEMA", "OUTPUT_SCHEMA"]
