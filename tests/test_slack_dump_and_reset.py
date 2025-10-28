from __future__ import annotations

import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import x_make_slack_dump_and_reset_z.x_cls_make_slack_dump_and_reset_x as module
from x_make_slack_dump_and_reset_z.x_cls_make_slack_dump_and_reset_x import (
    SCHEMA_VERSION,
    SlackChannelContext,
    SlackClientProtocol,
    SlackDumpAndReset,
    SlackFileRecord,
    SlackMessageRecord,
)


class FakeSlackClient(SlackClientProtocol):
    def __init__(self) -> None:
        self.downloaded: list[Path] = []
        self.deleted_messages: list[tuple[str, str]] = []
        self.deleted_files: list[str] = []

    def resolve_channel(self, identifier: str) -> SlackChannelContext:
        return SlackChannelContext(channel_id="C123", channel_name="general", messages=[])

    def fetch_messages(
        self,
        channel_id: str,
        *,
        include_threads: bool,
    ) -> list[SlackMessageRecord]:
        record = SlackMessageRecord(
            ts="123.456",
            text="Hello world",
            user="U123",
            raw={"ts": "123.456", "text": "Hello world"},
            files=[
                SlackFileRecord(
                    file_id="F123",
                    name="evidence.txt",
                    download_url="https://example.test/file",
                    mimetype="text/plain",
                )
            ],
            replies=[{"ts": "123.457", "text": "reply"}] if include_threads else [],
        )
        return [record]

    def download_file(self, file_record: SlackFileRecord, destination: Path) -> Path:
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / file_record.name
        path.write_text("evidence", encoding="utf-8")
        self.downloaded.append(path)
        return path

    def delete_message(self, channel_id: str, message_ts: str) -> None:
        self.deleted_messages.append((channel_id, message_ts))

    def delete_file(self, file_id: str) -> None:
        self.deleted_files.append(file_id)


def _make_runner(fake_client: FakeSlackClient) -> SlackDumpAndReset:
    return SlackDumpAndReset(client_factory=lambda token: fake_client, time_provider=lambda: datetime(2025, 10, 26, tzinfo=UTC))


def _build_payload(archive_root: Path) -> Mapping[str, Any]:
    return {
        "command": "x_make_slack_dump_and_reset_x",
        "parameters": {
            "slack_token": "xoxb-test",
            "channels": ["general"],
            "archive_root": str(archive_root),
        },
    }


def _prepare_archive_root(root: Path) -> Path:
    root.mkdir()
    older = root / "2025-10-20_Sprint"
    older.mkdir()
    os.utime(older, (older.stat().st_mtime - 100, older.stat().st_mtime - 100))
    newest = root / "2025-10-27_Sprint"
    newest.mkdir()
    return root


def test_run_exports_messages_and_deletes(tmp_path: Path) -> None:
    change_control = tmp_path / "Change Control"
    _prepare_archive_root(change_control)
    fake_client = FakeSlackClient()
    runner = _make_runner(fake_client)

    payload = _build_payload(change_control)
    result = runner.run(payload)

    assert result["status"] == "success"
    assert result["schema_version"] == SCHEMA_VERSION
    channels = result["channels"]
    assert isinstance(channels, list)
    assert channels
    channel_result = channels[0]
    assert isinstance(channel_result, dict)
    assert channel_result["channel_name"] == "general"
    assert channel_result["deleted"] is True
    assert channel_result["file_count"] == 1
    export_path = Path(channel_result["export_path"])
    assert export_path.exists()
    messages_file = export_path / "messages.json"
    assert messages_file.exists()
    messages = json.loads(messages_file.read_text(encoding="utf-8"))
    assert isinstance(messages, list)
    assert messages[0]["text"] == "Hello world"
    assert fake_client.downloaded
    assert fake_client.deleted_messages
    assert "F123" in fake_client.deleted_files


def test_run_uses_persistent_token_when_payload_missing(tmp_path: Path) -> None:
    change_control = tmp_path / "Change Control"
    _prepare_archive_root(change_control)
    fake_client = FakeSlackClient()
    runner = _make_runner(fake_client)

    payload = {
        "command": "x_make_slack_dump_and_reset_x",
        "parameters": {
            "channels": ["general"],
            "archive_root": str(change_control),
        },
    }

    original_env = os.environ.get("SLACK_TOKEN")
    os.environ["SLACK_TOKEN"] = "xoxe-ignored-refresh-token"
    original_resolver = module._resolve_persistent_slack_token
    try:
        module._resolve_persistent_slack_token = lambda: ("xoxp-from-vault", True)  # type: ignore[assignment]
        result = runner.run(payload)
    finally:
        if original_env is not None:
            os.environ["SLACK_TOKEN"] = original_env
        else:
            os.environ.pop("SLACK_TOKEN", None)
        module._resolve_persistent_slack_token = original_resolver  # type: ignore[assignment]

    assert result["status"] == "success"
