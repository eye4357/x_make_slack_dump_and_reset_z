from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pytest import MonkeyPatch  # noqa: PT013

from x_make_slack_dump_and_reset_z.x_cls_make_slack_dump_and_reset_x import (
    SCHEMA_VERSION,
    SlackChannelContext,
    SlackClientProtocol,
    SlackDumpAndReset,
    SlackFileRecord,
    SlackMessageRecord,
)


def expect(*, condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class FakeSlackClient(SlackClientProtocol):
    def __init__(self) -> None:
        self.downloaded: list[Path] = []
        self.deleted_messages: list[tuple[str, str]] = []
        self.deleted_files: list[str] = []

    def resolve_channel(self, identifier: str) -> SlackChannelContext:
        _ = identifier
        return SlackChannelContext(
            channel_id="C123", channel_name="general", messages=[]
        )

    def fetch_messages(
        self,
        channel_id: str,
        *,
        include_threads: bool,
    ) -> list[SlackMessageRecord]:
        _ = channel_id
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


def _make_runner(
    fake_client: FakeSlackClient,
    *,
    resolver: Callable[[], tuple[str | None, bool]] | None = None,
) -> SlackDumpAndReset:
    if resolver is None:
        return SlackDumpAndReset(
            client_factory=lambda _token: fake_client,
            time_provider=lambda: datetime(2025, 10, 26, tzinfo=UTC),
        )
    return SlackDumpAndReset(
        client_factory=lambda _token: fake_client,
        time_provider=lambda: datetime(2025, 10, 26, tzinfo=UTC),
        persistent_token_resolver=resolver,
    )


def _build_payload(archive_root: Path) -> dict[str, object]:
    return {
        "command": "x_make_slack_dump_and_reset_x",
        "parameters": {
            "slack_token": "slack-token",
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

    expect(
        condition=result["status"] == "success",
        message="Run should mark status as success",
    )
    expect(
        condition=result["schema_version"] == SCHEMA_VERSION,
        message="Schema version should match contract",
    )
    channels_obj = result["channels"]
    if not isinstance(channels_obj, list):
        message = "channels must be a list"
        raise TypeError(message)
    expect(condition=bool(channels_obj), message="channels result should not be empty")
    channel_mapping_raw = channels_obj[0]
    if not isinstance(channel_mapping_raw, Mapping):
        message = "channel entry must be mapping"
        raise TypeError(message)
    channel_mapping = cast("Mapping[str, object]", channel_mapping_raw)
    channel_data = dict(channel_mapping)
    expect(
        condition=channel_data.get("channel_name") == "general",
        message="Channel name should be general",
    )
    expect(
        condition=channel_data.get("deleted") is True,
        message="Channel should be marked deleted",
    )
    expect(
        condition=channel_data.get("file_count") == 1,
        message="File count should reflect exported files",
    )
    export_path_value = channel_data.get("export_path")
    if not isinstance(export_path_value, str):
        message = "export_path must be string"
        raise TypeError(message)
    export_path = Path(export_path_value)
    expect(
        condition=export_path.exists(),
        message="Export directory should exist",
    )
    messages_file = export_path / "messages.json"
    expect(
        condition=messages_file.exists(),
        message="messages.json should be written",
    )
    messages_raw: object = json.loads(messages_file.read_text(encoding="utf-8"))
    if not isinstance(messages_raw, list):
        message = "Messages must be list"
        raise TypeError(message)
    expect(condition=bool(messages_raw), message="Messages list should not be empty")
    first_message_raw = messages_raw[0]
    if not isinstance(first_message_raw, Mapping):
        message = "Message must be mapping"
        raise TypeError(message)
    first_message = dict(cast("Mapping[str, object]", first_message_raw))
    expect(
        condition=first_message.get("text") == "Hello world",
        message="Expected message text not found",
    )
    expect(
        condition=bool(fake_client.downloaded),
        message="Files should be downloaded",
    )
    expect(
        condition=bool(fake_client.deleted_messages),
        message="Messages should be deleted",
    )
    expect(
        condition="F123" in fake_client.deleted_files,
        message="File deletion should include F123",
    )


def test_run_uses_persistent_token_when_payload_missing(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    change_control = tmp_path / "Change Control"
    _prepare_archive_root(change_control)
    fake_client = FakeSlackClient()
    runner = _make_runner(
        fake_client,
        resolver=lambda: ("vault-token", True),
    )

    payload = {
        "command": "x_make_slack_dump_and_reset_x",
        "parameters": {
            "channels": ["general"],
            "archive_root": str(change_control),
        },
    }

    monkeypatch.setenv("SLACK_TOKEN", "placeholder-token")
    result = runner.run(payload)
    monkeypatch.delenv("SLACK_TOKEN", raising=False)

    expect(
        condition=result["status"] == "success",
        message="Persistent token should allow successful run",
    )
