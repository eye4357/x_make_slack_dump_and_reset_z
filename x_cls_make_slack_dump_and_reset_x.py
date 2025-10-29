"""Slack dump and reset workflow with JSON contracts."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
import time
from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from x_make_common_x.json_contracts import validate_payload

from x_make_slack_dump_and_reset_z.json_contracts import (
    ERROR_SCHEMA,
    INPUT_SCHEMA,
    OUTPUT_SCHEMA,
)

LOGGER = logging.getLogger(__name__)


class ResponseProtocol(Protocol):
    status_code: int
    headers: Mapping[str, str]

    def json(self) -> Any: ...

    def iter_content(self, chunk_size: int) -> Iterable[bytes]: ...

    def raise_for_status(self) -> None: ...


class SessionProtocol(Protocol):
    headers: MutableMapping[str, str]

    def request(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        json: Mapping[str, object] | None = None,
        stream: bool = False,
    ) -> ResponseProtocol: ...


class RequestsModule(Protocol):
    def Session(self) -> SessionProtocol: ...


if TYPE_CHECKING:
    requests: RequestsModule
else:  # pragma: no cover - import guard for runtime dependency
    try:
        requests = cast("RequestsModule", importlib.import_module("requests"))
    except ModuleNotFoundError as exc:  # pragma: no cover - surfaced at runtime
        message = "The 'requests' package is required for Slack exports"
        raise RuntimeError(message) from exc

SCHEMA_VERSION = "x_make_slack_dump_and_reset_x.run/1.0"
DEFAULT_EXPORT_SUBDIR = "slack_exports"
SLACK_API_ROOT = "https://slack.com/api"

__all__ = [
    "SCHEMA_VERSION",
    "SlackAPIError",
    "SlackChannelContext",
    "SlackDumpAndReset",
    "SlackFileRecord",
    "SlackMessageRecord",
    "SlackWebClient",
    "is_valid_slack_access_token",
]


class SlackAPIError(RuntimeError):
    """Raised when the Slack Web API returns an error response."""

    def __init__(
        self, method: str, error: str, payload: Mapping[str, object] | None = None
    ) -> None:
        message = f"Slack API call {method!r} failed: {error}"
        super().__init__(message)
        self.method = method
        self.error = error
        self.payload = payload


@dataclass(slots=True)
class SlackFileRecord:
    """Metadata about a Slack file to archive or delete."""

    file_id: str
    name: str
    download_url: str | None
    mimetype: str | None = None
    size: int | None = None


@dataclass(slots=True)
class SlackMessageRecord:
    """Representation of a Slack message including optional thread replies."""

    ts: str
    text: str
    user: str | None
    raw: dict[str, Any]
    files: list[SlackFileRecord] = field(default_factory=list)
    replies: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class SlackChannelContext:
    """Context captured during export for a single Slack channel."""

    channel_id: str
    channel_name: str
    messages: list[SlackMessageRecord]


class SlackClientProtocol(Protocol):
    """Subset of Slack client behaviour required by the exporter."""

    def resolve_channel(self, identifier: str) -> SlackChannelContext: ...

    def fetch_messages(
        self,
        channel_id: str,
        *,
        include_threads: bool,
    ) -> list[SlackMessageRecord]: ...

    def download_file(
        self, file_record: SlackFileRecord, destination: Path
    ) -> Path: ...

    def delete_message(self, channel_id: str, message_ts: str) -> None: ...

    def delete_file(self, file_id: str) -> None: ...


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def is_valid_slack_access_token(token: str) -> bool:
    """Return True when the token looks like a usable Slack access token."""

    if not token:
        return False
    normalized = token.strip()
    if not normalized:
        return False
    return not normalized.startswith(("xoxe-", "xoxr-"))


def _resolve_persistent_slack_token() -> tuple[str | None, bool]:
    """Return a Slack token from the persistent vault when available."""

    try:
        from x_make_persistent_env_var_x.x_cls_make_persistent_env_var_x import (
            x_cls_make_persistent_env_var_x,
        )
    except Exception:  # pragma: no cover - optional dependency at runtime
        return None, False

    try:
        reader = x_cls_make_persistent_env_var_x("SLACK_TOKEN", quiet=True)
        persisted = reader.get_user_env()
    except Exception:
        return None, False
    if isinstance(persisted, str) and persisted.strip():
        return persisted.strip(), True
    return None, False


class SlackWebClient:
    """Thin wrapper around the Slack Web API using requests."""

    def __init__(
        self,
        token: str,
        *,
        session: SessionProtocol | None = None,
        sleeper: Callable[[float], None] = _sleep,
    ) -> None:
        self._session: SessionProtocol = session or requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=utf-8",
            }
        )
        self._sleeper = sleeper
        self._channel_cache: dict[str, dict[str, Any]] = {}
        self._channel_name_to_id: dict[str, str] = {}

    def resolve_channel(self, identifier: str) -> SlackChannelContext:
        identifier = identifier.removeprefix("#")
        channel_payload = self._resolve_channel_payload(identifier)
        channel_id = str(channel_payload["id"])
        channel_name = str(channel_payload.get("name", channel_id))
        return SlackChannelContext(
            channel_id=channel_id, channel_name=channel_name, messages=[]
        )

    def iter_channels(self) -> Iterable[dict[str, Any]]:
        """Yield channel payloads from the Slack API."""

        yield from self._iterate_channels()

    def fetch_messages(
        self,
        channel_id: str,
        *,
        include_threads: bool,
    ) -> list[SlackMessageRecord]:
        messages: list[SlackMessageRecord] = []
        cursor: str | None = None
        while True:
            payload = self._api_call(
                "conversations.history",
                params={"channel": channel_id, "cursor": cursor, "limit": 200},
            )
            raw_messages = payload.get("messages", [])
            if not isinstance(raw_messages, list):
                raise SlackAPIError(
                    "conversations.history", "invalid_messages_payload", payload
                )
            for raw in raw_messages:
                if not isinstance(raw, dict):
                    continue
                record = self._build_message_record(channel_id, raw, include_threads)
                messages.append(record)
            cursor = self._next_cursor(payload)
            if not cursor:
                break
        return messages

    def download_file(self, file_record: SlackFileRecord, destination: Path) -> Path:
        destination.mkdir(parents=True, exist_ok=True)
        if not file_record.download_url:
            raise SlackAPIError(
                "files.download",
                "missing_download_url",
                {"file": file_record.file_id},
            )
        response = self._http_request("GET", file_record.download_url, stream=True)
        target_path = destination / Path(file_record.name).name
        with target_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    handle.write(chunk)
        return target_path

    def delete_message(self, channel_id: str, message_ts: str) -> None:
        self._api_call(
            "chat.delete",
            http_method="POST",
            json_payload={"channel": channel_id, "ts": message_ts},
        )

    def delete_file(self, file_id: str) -> None:
        self._api_call(
            "files.delete",
            http_method="POST",
            json_payload={"file": file_id},
        )

    # --- internal helpers -------------------------------------------------

    def _resolve_channel_payload(self, identifier: str) -> dict[str, Any]:
        if identifier in self._channel_cache:
            return self._channel_cache[identifier]
        if identifier in self._channel_name_to_id:
            cached_id = self._channel_name_to_id[identifier]
            return self._channel_cache[cached_id]

        for payload in self._iterate_channels():
            channel_id = str(payload["id"])
            name = str(payload.get("name", channel_id))
            self._channel_cache[channel_id] = payload
            self._channel_name_to_id[name] = channel_id
            if channel_id == identifier or name == identifier:
                return payload
        raise SlackAPIError(
            "conversations.list", "channel_not_found", {"query": identifier}
        )

    def _iterate_channels(self) -> Iterable[dict[str, Any]]:
        cursor: str | None = None
        while True:
            payload = self._api_call(
                "conversations.list",
                params={"exclude_archived": True, "cursor": cursor, "limit": 200},
            )
            channels = payload.get("channels", [])
            if not isinstance(channels, list):
                raise SlackAPIError(
                    "conversations.list", "invalid_channels_payload", payload
                )
            for channel in channels:
                if isinstance(channel, dict):
                    yield channel
            cursor = self._next_cursor(payload)
            if not cursor:
                break

    def _build_message_record(
        self,
        channel_id: str,
        raw: dict[str, Any],
        include_threads: bool,
    ) -> SlackMessageRecord:
        text = str(raw.get("text", ""))
        user = raw.get("user")
        files_payload = raw.get("files", [])
        files: list[SlackFileRecord] = []
        if isinstance(files_payload, list):
            for file_item in files_payload:
                if not isinstance(file_item, dict):
                    continue
                file_id = str(file_item.get("id", ""))
                if not file_id:
                    continue
                file_record = SlackFileRecord(
                    file_id=file_id,
                    name=str(file_item.get("name", file_id)),
                    download_url=file_item.get("url_private_download")
                    or file_item.get("url_private"),
                    mimetype=file_item.get("mimetype"),
                    size=file_item.get("size"),
                )
                files.append(file_record)
        record = SlackMessageRecord(
            ts=str(raw.get("ts", "")),
            text=text,
            user=str(user) if isinstance(user, str) else None,
            raw=raw,
            files=files,
        )
        if include_threads and raw.get("reply_count"):
            replies_payload = self._api_call(
                "conversations.replies",
                params={"channel": channel_id, "ts": record.ts, "limit": 200},
            )
            replies = replies_payload.get("messages", [])
            if isinstance(replies, list):
                for reply in replies:
                    if not isinstance(reply, dict):
                        continue
                    if reply.get("ts") == record.ts:
                        continue  # skip parent repeats
                    record.replies.append(reply)
        return record

    def _api_call(
        self,
        method: str,
        *,
        params: Mapping[str, object] | None = None,
        json_payload: Mapping[str, object] | None = None,
        http_method: str | None = None,
        **_ignored: object,
    ) -> dict[str, Any]:
        params_dict = dict(params) if params else None
        json_dict = dict(json_payload) if json_payload else None
        inferred_method = http_method or ("POST" if json_dict is not None else "GET")
        response = self._http_request(
            inferred_method,
            f"{SLACK_API_ROOT}/{method}",
            params=params_dict,
            json=json_dict,
        )
        payload = response.json()
        if not isinstance(payload, dict):
            raise SlackAPIError(method, "invalid_payload", {})
        if not payload.get("ok", False):
            error_text = str(payload.get("error", "unknown_error"))
            raise SlackAPIError(method, error_text, payload)
        return payload

    def _http_request(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, object] | None = None,
        json: Mapping[str, object] | None = None,
        stream: bool = False,
    ) -> ResponseProtocol:
        backoff = 1.0
        while True:
            params_dict = None
            if params:
                params_dict = {
                    str(key): str(value)
                    for key, value in params.items()
                    if value is not None
                }
            json_dict = dict(json) if json else None
            response = self._session.request(
                method,
                url,
                params=params_dict,
                json=json_dict,
                stream=stream,
            )
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                sleep_for = float(retry_after) if retry_after else backoff
                LOGGER.debug("Slack rate limit hit; sleeping for %s seconds", sleep_for)
                self._sleeper(sleep_for)
                backoff = min(backoff * 2.0, 30.0)
                continue
            response.raise_for_status()
            return response

    @staticmethod
    def _next_cursor(payload: Mapping[str, Any]) -> str | None:
        metadata = payload.get("response_metadata")
        if isinstance(metadata, Mapping):
            cursor = metadata.get("next_cursor")
            if isinstance(cursor, str) and cursor:
                return cursor
        return None


@dataclass(slots=True)
class SlackDumpParameters:
    slack_token: str
    channels: Sequence[str | Mapping[str, object]]
    archive_root: Path
    delete_after_export: bool
    include_files: bool
    include_threads: bool
    dry_run: bool
    skip_channels: set[str]
    notes: Sequence[str]


class SlackDumpAndReset:
    """Export Slack channel history to Change Control and optionally purge."""

    def __init__(
        self,
        client_factory: Callable[[str], SlackClientProtocol] | None = None,
        *,
        time_provider: Callable[[], datetime] = _now_utc,
    ) -> None:
        self._client_factory = client_factory
        self._time_provider = time_provider

    def run(self, payload: Mapping[str, object]) -> dict[str, object]:
        validate_payload(payload, INPUT_SCHEMA)
        parameters = self._parse_parameters(payload)
        export_root = self._resolve_export_root(parameters.archive_root)
        timestamp = self._time_provider().strftime("%Y%m%dT%H%M%SZ")
        export_folder = export_root / DEFAULT_EXPORT_SUBDIR / timestamp
        export_folder.mkdir(parents=True, exist_ok=True)

        client = self._create_client(parameters.slack_token)

        results: list[dict[str, object]] = []
        info_messages: list[str] = []

        for channel_spec in parameters.channels:
            channel_identifier, label = self._normalise_channel_identifier(channel_spec)
            if (
                channel_identifier in parameters.skip_channels
                or label in parameters.skip_channels
            ):
                info_messages.append(f"Skipped channel {label} via configuration")
                continue
            context = client.resolve_channel(channel_identifier)
            messages = client.fetch_messages(
                context.channel_id, include_threads=parameters.include_threads
            )
            context.messages = messages
            channel_dir = export_folder / context.channel_name
            channel_dir.mkdir(parents=True, exist_ok=True)
            message_path = channel_dir / "messages.json"
            with message_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    [self._serialise_message(record) for record in context.messages],
                    handle,
                    indent=2,
                    ensure_ascii=False,
                )
            files_dir = channel_dir / "files"
            expected_files = sum(len(message.files) for message in context.messages)
            downloaded_files = 0
            if parameters.include_files and expected_files:
                for message in context.messages:
                    for file_record in message.files:
                        if parameters.dry_run:
                            downloaded_files += 1
                            continue
                        try:
                            client.download_file(file_record, files_dir)
                            downloaded_files += 1
                        except SlackAPIError as exc:
                            LOGGER.warning(
                                "Failed to download file %s: %s",
                                file_record.file_id,
                                exc,
                            )
                            info_messages.append(
                                f"File download failed for {file_record.file_id} in {context.channel_name}: {exc.error}"
                            )
            deleted = False
            delete_failures = False
            if parameters.delete_after_export and not parameters.dry_run:
                for message in context.messages:
                    try:
                        client.delete_message(context.channel_id, message.ts)
                        for reply in message.replies:
                            reply_ts = str(reply.get("ts", ""))
                            if reply_ts:
                                client.delete_message(context.channel_id, reply_ts)
                        for file_record in message.files:
                            try:
                                client.delete_file(file_record.file_id)
                            except SlackAPIError as exc:
                                LOGGER.debug(
                                    "Failed to delete file %s: %s",
                                    file_record.file_id,
                                    exc,
                                )
                                info_messages.append(
                                    f"File delete failed for {file_record.file_id} in {context.channel_name}: {exc.error}"
                                )
                                delete_failures = True
                    except SlackAPIError as exc:
                        LOGGER.warning(
                            "Failed to delete message %s in channel %s: %s",
                            message.ts,
                            context.channel_name,
                            exc,
                        )
                        info_messages.append(
                            f"Message delete failed for channel {context.channel_name} ts={message.ts}: {exc.error}"
                        )
                        delete_failures = True
                deleted = not delete_failures
            results.append(
                {
                    "channel_id": context.channel_id,
                    "channel_name": context.channel_name,
                    "message_count": sum(
                        1 + len(msg.replies) for msg in context.messages
                    ),
                    "file_count": (
                        downloaded_files if parameters.include_files else expected_files
                    ),
                    "export_path": str(channel_dir.as_posix()),
                    "deleted": deleted,
                }
            )

        output: dict[str, object] = {
            "status": "success",
            "schema_version": SCHEMA_VERSION,
            "export_root": export_folder.as_posix(),
            "channels": results,
        }
        if info_messages or parameters.notes:
            combined_messages: list[str] = list(parameters.notes)
            combined_messages.extend(info_messages)
            output["messages"] = combined_messages
        validate_payload(output, OUTPUT_SCHEMA)
        return output

    def _create_client(self, token: str) -> SlackClientProtocol:
        factory = self._client_factory or (lambda t: SlackWebClient(t))
        return factory(token)

    def _parse_parameters(self, payload: Mapping[str, object]) -> SlackDumpParameters:
        parameters_raw = payload["parameters"]
        assert isinstance(parameters_raw, Mapping)
        token_obj = parameters_raw.get("slack_token")
        token = token_obj if isinstance(token_obj, str) and token_obj else None
        if token is None:
            env_value = os.getenv("SLACK_TOKEN")
            if isinstance(env_value, str) and env_value.strip():
                candidate = env_value.strip()
                if is_valid_slack_access_token(candidate):
                    token = candidate
        if token is None:
            persisted_token, _ = _resolve_persistent_slack_token()
            if persisted_token:
                token = persisted_token
        elif not is_valid_slack_access_token(token):
            persisted_token, _ = _resolve_persistent_slack_token()
            if persisted_token and is_valid_slack_access_token(persisted_token):
                token = persisted_token
        if not isinstance(token, str) or not token:
            raise RuntimeError(
                "Slack token not provided in payload or SLACK_TOKEN environment variable"
            )
        archive_root_raw = parameters_raw.get("archive_root")
        if not isinstance(archive_root_raw, str) or not archive_root_raw:
            raise RuntimeError("archive_root must be a non-empty string path")
        channels_raw = parameters_raw.get("channels")
        if not isinstance(channels_raw, Sequence) or not channels_raw:
            raise RuntimeError("channels must be a non-empty array")
        channels: list[str | Mapping[str, object]] = []
        for item in channels_raw:
            if (isinstance(item, str) and item) or isinstance(item, Mapping):
                channels.append(item)
            else:
                raise RuntimeError(
                    "channels entries must be strings or objects with id/name"
                )
        skip_raw = parameters_raw.get("skip_channels")
        skip_channels: set[str] = set()
        if isinstance(skip_raw, Sequence):
            for item in skip_raw:
                if isinstance(item, str) and item:
                    skip_channels.add(item)
        delete_after_export = bool(parameters_raw.get("delete_after_export", True))
        include_files = bool(parameters_raw.get("include_files", True))
        include_threads = bool(parameters_raw.get("include_threads", True))
        dry_run = bool(parameters_raw.get("dry_run", False))
        notes_raw = parameters_raw.get("notes")
        notes: list[str] = []
        if isinstance(notes_raw, Sequence):
            for note in notes_raw:
                if isinstance(note, str):
                    notes.append(note)
        return SlackDumpParameters(
            slack_token=token,
            channels=channels,
            archive_root=Path(archive_root_raw).expanduser().resolve(),
            delete_after_export=delete_after_export,
            include_files=include_files,
            include_threads=include_threads,
            dry_run=dry_run,
            skip_channels=skip_channels,
            notes=notes,
        )

    def _resolve_export_root(self, archive_root: Path) -> Path:
        if not archive_root.exists():
            raise FileNotFoundError(f"Archive root does not exist: {archive_root}")
        subdirectories = [item for item in archive_root.iterdir() if item.is_dir()]
        if not subdirectories:
            raise FileNotFoundError(
                f"Archive root {archive_root} has no subdirectories to target"
            )
        latest_directory = max(subdirectories, key=lambda item: item.stat().st_mtime)
        return latest_directory

    @staticmethod
    def _normalise_channel_identifier(
        channel_spec: str | Mapping[str, object],
    ) -> tuple[str, str]:
        if isinstance(channel_spec, Mapping):
            channel_id = channel_spec.get("id")
            channel_name = channel_spec.get("name")
            if isinstance(channel_id, str) and channel_id:
                label = (
                    channel_name
                    if isinstance(channel_name, str) and channel_name
                    else channel_id
                )
                return channel_id, label
            if isinstance(channel_name, str) and channel_name:
                return channel_name, channel_name
            raise RuntimeError("Channel mapping must provide 'id' or 'name'")
        if isinstance(channel_spec, str) and channel_spec:
            return channel_spec, channel_spec.lstrip("#")
        raise RuntimeError(
            "Channel specification must be a non-empty string or mapping"
        )

    @staticmethod
    def _serialise_message(record: SlackMessageRecord) -> dict[str, Any]:
        data: dict[str, Any] = {
            "ts": record.ts,
            "text": record.text,
            "user": record.user,
            "raw": record.raw,
        }
        if record.files:
            data["files"] = [
                {
                    "file_id": file_record.file_id,
                    "name": file_record.name,
                    "mimetype": file_record.mimetype,
                    "size": file_record.size,
                }
                for file_record in record.files
            ]
        if record.replies:
            data["replies"] = record.replies
        return data


def _load_json_source(path: str | None) -> Mapping[str, object]:
    if path is None or path == "-":
        payload = json.load(sys.stdin)
    else:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise RuntimeError("Input payload must be a JSON object")
    return payload


def _dump_json(output: Mapping[str, object]) -> None:
    json.dump(output, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


def _prompt_delete_confirmation(payload: Mapping[str, object]) -> None:
    parameters_obj = payload.get("parameters")
    if not isinstance(parameters_obj, dict):
        return
    delete_after = parameters_obj.get("delete_after_export")
    delete_enabled = True if delete_after is None else bool(delete_after)
    if not delete_enabled:
        return
    if bool(parameters_obj.get("dry_run", False)):
        return

    while True:
        response = (
            input(
                "Archive captured. Delete Slack messages and files after export? [y/N]: "
            )
            .strip()
            .lower()
        )
        if response in {"y", "yes"}:
            print(
                "Confirmed. Slack source will be purged post-export.", file=sys.stderr
            )
            return
        if response in {"", "n", "no"}:
            parameters_obj["delete_after_export"] = False
            print(
                "Deletion skipped. Slack history remains intact for aggregation.",
                file=sys.stderr,
            )
            return
        print("Please respond with 'y' or 'n'.", file=sys.stderr)


def _run_cli() -> int:
    parser = argparse.ArgumentParser(
        prog="x_make_slack_dump_and_reset_z",
        description="Export and reset Slack channels using JSON contracts.",
    )
    parser.add_argument("--input", help="Path to JSON payload (default: stdin)")
    parser.add_argument(
        "--output", help="File path to write JSON response (default: stdout)"
    )
    args = parser.parse_args()

    try:
        payload = _load_json_source(args.input)
        _prompt_delete_confirmation(payload)
        runner = SlackDumpAndReset()
        output = runner.run(payload)
    except Exception as exc:
        LOGGER.exception("Slack dump run failed")
        error_payload = {
            "status": "failure",
            "message": str(exc),
            "details": {"type": exc.__class__.__name__},
        }
        try:
            validate_payload(error_payload, ERROR_SCHEMA)
        except Exception:  # noqa: BLE001 - best effort to preserve error output
            pass
        if args.output:
            with open(args.output, "w", encoding="utf-8") as handle:
                json.dump(error_payload, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
        else:
            _dump_json(error_payload)
        return 1

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    else:
        _dump_json(output)
    return 0


if __name__ == "__main__":
    sys.exit(_run_cli())
