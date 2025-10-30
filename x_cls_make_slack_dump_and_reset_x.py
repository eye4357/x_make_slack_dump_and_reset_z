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
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeAlias, cast

from x_make_common_x.json_contracts import validate_payload

from x_make_slack_dump_and_reset_z.json_contracts import (
    ERROR_SCHEMA,
    INPUT_SCHEMA,
    OUTPUT_SCHEMA,
)

LOGGER = logging.getLogger(__name__)

HTTP_TOO_MANY_REQUESTS = 429
JSON_KEY_ERROR_MESSAGE = "JSON object keys must be strings"
JSON_OBJECT_REQUIRED_MESSAGE = "Expected JSON object"
UNSUPPORTED_JSON_VALUE_MESSAGE = "Unsupported JSON value encountered"


JSONPrimitive: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONPrimitive | list["JSONValue"] | dict[str, "JSONValue"]
JSONObject: TypeAlias = dict[str, JSONValue]


def _coerce_json_value(value: object) -> JSONValue:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_coerce_json_value(item) for item in value]
    if isinstance(value, dict):
        result: dict[str, JSONValue] = {}
        for key, inner in value.items():
            if not isinstance(key, str):
                raise TypeError(JSON_KEY_ERROR_MESSAGE)
            result[key] = _coerce_json_value(inner)
        return result
    raise TypeError(UNSUPPORTED_JSON_VALUE_MESSAGE)


def _coerce_json_object(value: object) -> JSONObject:
    if not isinstance(value, dict):
        raise TypeError(JSON_OBJECT_REQUIRED_MESSAGE)
    result: JSONObject = {}
    for key, inner in value.items():
        if not isinstance(key, str):
            raise TypeError(JSON_KEY_ERROR_MESSAGE)
        result[key] = _coerce_json_value(inner)
    return result


def _maybe_json_object(value: object) -> JSONObject | None:
    try:
        return _coerce_json_object(value)
    except TypeError:
        return None


class ResponseProtocol(Protocol):
    status_code: int
    headers: Mapping[str, str]

    def json(self) -> JSONValue: ...

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
    Session: Callable[[], SessionProtocol]


class PersistentEnvReaderProtocol(Protocol):
    def get_user_env(self) -> str | None: ...


class PersistentEnvReaderFactoryProtocol(Protocol):
    def __call__(
        self,
        var: str = "",
        value: str = "",
        *,
        quiet: bool = False,
        ctx: object | None = None,
        **token_options: object,
    ) -> PersistentEnvReaderProtocol: ...


if TYPE_CHECKING:
    requests: RequestsModule
    from x_make_persistent_env_var_x.x_cls_make_persistent_env_var_x import (  # type: ignore[import-not-found]
        x_cls_make_persistent_env_var_x as imported_persistent_env_factory,
    )

    PersistentEnvReaderFactory: PersistentEnvReaderFactoryProtocol | None = cast(
        "PersistentEnvReaderFactoryProtocol", imported_persistent_env_factory
    )
else:  # pragma: no cover - import guard for runtime dependency
    try:
        requests = cast("RequestsModule", importlib.import_module("requests"))
    except ModuleNotFoundError as exc:  # pragma: no cover - surfaced at runtime
        message = "The 'requests' package is required for Slack exports"
        raise RuntimeError(message) from exc
    try:
        from x_make_persistent_env_var_x.x_cls_make_persistent_env_var_x import (
            x_cls_make_persistent_env_var_x as imported_persistent_env_factory,
        )

        PersistentEnvReaderFactory = cast(
            "PersistentEnvReaderFactoryProtocol", imported_persistent_env_factory
        )
    except ImportError:  # pragma: no cover - optional dependency resolved at runtime
        PersistentEnvReaderFactory = None

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
    raw: JSONObject
    files: list[SlackFileRecord] = field(default_factory=list)
    replies: list[JSONObject] = field(default_factory=list)


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

    if PersistentEnvReaderFactory is None:
        return None, False

    reader = PersistentEnvReaderFactory("SLACK_TOKEN", quiet=True)
    persisted: str | None = None
    with suppress(Exception):
        persisted = reader.get_user_env()
    if isinstance(persisted, str):
        candidate = persisted.strip()
        if candidate:
            return candidate, True
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
        self._channel_cache: dict[str, JSONObject] = {}
        self._channel_name_to_id: dict[str, str] = {}

    def resolve_channel(self, identifier: str) -> SlackChannelContext:
        identifier = identifier.removeprefix("#")
        channel_payload = self._resolve_channel_payload(identifier)
        channel_id = str(channel_payload["id"])
        channel_name = str(channel_payload.get("name", channel_id))
        return SlackChannelContext(
            channel_id=channel_id, channel_name=channel_name, messages=[]
        )

    def iter_channels(self) -> Iterable[JSONObject]:
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
                method = "conversations.history"
                error = "invalid_messages_payload"
                raise SlackAPIError(method, error, payload)
            for raw in raw_messages:
                message_obj = _maybe_json_object(raw)
                if message_obj is None:
                    continue
                record = self._build_message_record(
                    channel_id,
                    message_obj,
                    include_threads=include_threads,
                )
                messages.append(record)
            cursor = self._next_cursor(payload)
            if not cursor:
                break
        return messages

    def download_file(self, file_record: SlackFileRecord, destination: Path) -> Path:
        destination.mkdir(parents=True, exist_ok=True)
        if not file_record.download_url:
            method = "files.download"
            error = "missing_download_url"
            details = {"file": file_record.file_id}
            raise SlackAPIError(method, error, details)
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

    def _resolve_channel_payload(self, identifier: str) -> JSONObject:
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
            if identifier in {channel_id, name}:
                return payload
        method = "conversations.list"
        error = "channel_not_found"
        details = {"query": identifier}
        raise SlackAPIError(method, error, details)

    def _iterate_channels(self) -> Iterable[JSONObject]:
        cursor: str | None = None
        while True:
            payload = self._api_call(
                "conversations.list",
                params={"exclude_archived": True, "cursor": cursor, "limit": 200},
            )
            channels = payload.get("channels", [])
            if not isinstance(channels, list):
                method = "conversations.list"
                error = "invalid_channels_payload"
                raise SlackAPIError(method, error, payload)
            for channel in channels:
                channel_obj = _maybe_json_object(channel)
                if channel_obj is not None:
                    yield channel_obj
            cursor = self._next_cursor(payload)
            if not cursor:
                break

    def _build_message_record(
        self,
        channel_id: str,
        raw: JSONObject,
        *,
        include_threads: bool,
    ) -> SlackMessageRecord:
        text = str(raw.get("text", ""))
        user = raw.get("user")
        files_payload = raw.get("files", [])
        files: list[SlackFileRecord] = []
        if isinstance(files_payload, list):
            for file_item in files_payload:
                file_obj = _maybe_json_object(file_item)
                if file_obj is None:
                    continue
                file_id = str(file_obj.get("id", ""))
                if not file_id:
                    continue
                download_candidate = file_obj.get(
                    "url_private_download"
                ) or file_obj.get("url_private")
                download_url = (
                    str(download_candidate)
                    if isinstance(download_candidate, str)
                    else None
                )
                mimetype_obj = file_obj.get("mimetype")
                mimetype = str(mimetype_obj) if isinstance(mimetype_obj, str) else None
                size_obj = file_obj.get("size")
                size = size_obj if isinstance(size_obj, int) else None
                file_record = SlackFileRecord(
                    file_id=file_id,
                    name=str(file_obj.get("name", file_id)),
                    download_url=download_url,
                    mimetype=mimetype,
                    size=size,
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
                    reply_obj = _maybe_json_object(reply)
                    if reply_obj is None:
                        continue
                    if reply_obj.get("ts") == record.ts:
                        continue  # skip parent repeats
                    record.replies.append(reply_obj)
        return record

    def _api_call(
        self,
        method: str,
        *,
        params: Mapping[str, object] | None = None,
        json_payload: Mapping[str, object] | None = None,
        http_method: str | None = None,
        **_ignored: object,
    ) -> JSONObject:
        params_dict = dict(params) if params else None
        json_dict = dict(json_payload) if json_payload else None
        inferred_method = http_method or ("POST" if json_dict is not None else "GET")
        response = self._http_request(
            inferred_method,
            f"{SLACK_API_ROOT}/{method}",
            params=params_dict,
            json=json_dict,
        )
        payload_raw = response.json()
        payload = _maybe_json_object(payload_raw)
        if payload is None:
            error = "invalid_payload"
            raise SlackAPIError(method, error, {})
        ok_value = payload.get("ok")
        is_ok = bool(ok_value) if isinstance(ok_value, bool) else False
        if not is_ok:
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
            if response.status_code == HTTP_TOO_MANY_REQUESTS:
                retry_after = response.headers.get("Retry-After")
                sleep_for = float(retry_after) if retry_after else backoff
                LOGGER.debug("Slack rate limit hit; sleeping for %s seconds", sleep_for)
                self._sleeper(sleep_for)
                backoff = min(backoff * 2.0, 30.0)
                continue
            response.raise_for_status()
            return response

    @staticmethod
    def _next_cursor(payload: Mapping[str, JSONValue]) -> str | None:
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
        persistent_token_resolver: Callable[[], tuple[str | None, bool]] = (
            _resolve_persistent_slack_token
        ),
    ) -> None:
        self._client_factory = client_factory
        self._time_provider = time_provider
        self._persistent_token_resolver = persistent_token_resolver

    def run(self, payload: Mapping[str, object]) -> dict[str, object]:
        validate_payload(payload, INPUT_SCHEMA)
        parameters = self._parse_parameters(payload)
        export_folder = self._prepare_export_folder(parameters.archive_root)
        client = self._create_client(parameters.slack_token)
        channel_results, info_messages = self._process_channels(
            client, parameters, export_folder
        )
        output = self._build_output(
            export_folder, channel_results, info_messages, parameters.notes
        )
        validate_payload(output, OUTPUT_SCHEMA)
        return output

    def _prepare_export_folder(self, archive_root: Path) -> Path:
        export_root = self._resolve_export_root(archive_root)
        timestamp = self._time_provider().strftime("%Y%m%dT%H%M%SZ")
        export_folder = export_root / DEFAULT_EXPORT_SUBDIR / timestamp
        export_folder.mkdir(parents=True, exist_ok=True)
        return export_folder

    def _process_channels(
        self,
        client: SlackClientProtocol,
        parameters: SlackDumpParameters,
        export_folder: Path,
    ) -> tuple[list[dict[str, object]], list[str]]:
        results: list[dict[str, object]] = []
        info_messages: list[str] = []
        normalised_channels = self._normalise_channels(parameters.channels)
        for channel_identifier, label in normalised_channels:
            if self._should_skip_channel(
                channel_identifier, label, parameters.skip_channels
            ):
                message = f"Skipped channel {label} via configuration"
                info_messages.append(message)
                continue
            channel_result, channel_messages = self._export_channel(
                client,
                channel_identifier,
                export_folder,
                parameters,
            )
            results.append(channel_result)
            info_messages.extend(channel_messages)
        return results, info_messages

    def _normalise_channels(
        self, channels: Sequence[str | Mapping[str, object]]
    ) -> list[tuple[str, str]]:
        return [
            self._normalise_channel_identifier(channel_spec)
            for channel_spec in channels
        ]

    def _export_channel(
        self,
        client: SlackClientProtocol,
        channel_identifier: str,
        export_folder: Path,
        parameters: SlackDumpParameters,
    ) -> tuple[dict[str, object], list[str]]:
        context = client.resolve_channel(channel_identifier)
        messages = client.fetch_messages(
            context.channel_id, include_threads=parameters.include_threads
        )
        context.messages = messages
        channel_dir = export_folder / context.channel_name
        channel_dir.mkdir(parents=True, exist_ok=True)
        self._write_messages_file(channel_dir, context.messages)
        expected_files = sum(len(message.files) for message in context.messages)
        downloaded_files, download_messages = self._download_channel_files(
            client,
            context,
            channel_dir=channel_dir,
            include_files=parameters.include_files,
            dry_run=parameters.dry_run,
        )
        deleted, delete_messages = self._maybe_delete_history(
            client,
            context,
            delete_after_export=parameters.delete_after_export,
            dry_run=parameters.dry_run,
        )
        channel_messages = [*download_messages, *delete_messages]
        message_count = sum(1 + len(msg.replies) for msg in context.messages)
        file_count = downloaded_files if parameters.include_files else expected_files
        result = {
            "channel_id": context.channel_id,
            "channel_name": context.channel_name,
            "message_count": message_count,
            "file_count": file_count,
            "export_path": channel_dir.as_posix(),
            "deleted": deleted,
        }
        return result, channel_messages

    @staticmethod
    def _should_skip_channel(
        channel_identifier: str, label: str, skip_channels: set[str]
    ) -> bool:
        return channel_identifier in skip_channels or label in skip_channels

    def _write_messages_file(
        self, channel_dir: Path, messages: Sequence[SlackMessageRecord]
    ) -> None:
        message_path = channel_dir / "messages.json"
        with message_path.open("w", encoding="utf-8") as handle:
            json.dump(
                [self._serialise_message(record) for record in messages],
                handle,
                indent=2,
                ensure_ascii=False,
            )

    def _download_channel_files(
        self,
        client: SlackClientProtocol,
        context: SlackChannelContext,
        *,
        channel_dir: Path,
        include_files: bool,
        dry_run: bool,
    ) -> tuple[int, list[str]]:
        info_messages: list[str] = []
        downloaded_files = 0
        if not include_files:
            return downloaded_files, info_messages
        files_dir = channel_dir / "files"
        has_files = any(message.files for message in context.messages)
        if not has_files:
            return downloaded_files, info_messages
        for message in context.messages:
            for file_record in message.files:
                if dry_run:
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
                    error_message = (
                        "File download failed for "
                        f"{file_record.file_id} in {context.channel_name}: {exc.error}"
                    )
                    info_messages.append(error_message)
        return downloaded_files, info_messages

    def _maybe_delete_history(
        self,
        client: SlackClientProtocol,
        context: SlackChannelContext,
        *,
        delete_after_export: bool,
        dry_run: bool,
    ) -> tuple[bool, list[str]]:
        info_messages: list[str] = []
        if not delete_after_export or dry_run:
            return False, info_messages
        delete_failures = False
        for message in context.messages:
            message_failed, message_info = self._delete_message_and_files(
                client, context, message
            )
            info_messages.extend(message_info)
            if message_failed:
                delete_failures = True
        return (not delete_failures), info_messages

    def _delete_message_and_files(
        self,
        client: SlackClientProtocol,
        context: SlackChannelContext,
        message: SlackMessageRecord,
    ) -> tuple[bool, list[str]]:
        info_messages: list[str] = []
        try:
            client.delete_message(context.channel_id, message.ts)
            for reply in message.replies:
                reply_ts = str(reply.get("ts", ""))
                if reply_ts:
                    client.delete_message(context.channel_id, reply_ts)
        except SlackAPIError as exc:
            LOGGER.warning(
                "Failed to delete message %s in channel %s: %s",
                message.ts,
                context.channel_name,
                exc,
            )
            failure_message = (
                "Message delete failed for channel "
                f"{context.channel_name} ts={message.ts}: {exc.error}"
            )
            info_messages.append(failure_message)
            return True, info_messages

        message_failed = False
        for file_record in message.files:
            try:
                client.delete_file(file_record.file_id)
            except SlackAPIError as exc:
                LOGGER.debug(
                    "Failed to delete file %s: %s",
                    file_record.file_id,
                    exc,
                )
                error_message = (
                    "File delete failed for "
                    f"{file_record.file_id} in {context.channel_name}: {exc.error}"
                )
                info_messages.append(error_message)
                message_failed = True
        return message_failed, info_messages

    def _build_output(
        self,
        export_folder: Path,
        results: Sequence[dict[str, object]],
        info_messages: Sequence[str],
        notes: Sequence[str],
    ) -> dict[str, object]:
        output: dict[str, object] = {
            "status": "success",
            "schema_version": SCHEMA_VERSION,
            "export_root": export_folder.as_posix(),
            "channels": list(results),
        }
        if info_messages or notes:
            combined_messages = [*notes, *info_messages]
            output["messages"] = combined_messages
        return output

    def _create_client(self, token: str) -> SlackClientProtocol:
        if self._client_factory is not None:
            return self._client_factory(token)
        return SlackWebClient(token)

    def _parse_parameters(self, payload: Mapping[str, object]) -> SlackDumpParameters:
        parameters_raw = self._extract_parameters(payload)
        token = self._resolve_token(parameters_raw)
        archive_root = self._parse_archive_root(parameters_raw)
        channels = self._parse_channels(parameters_raw)
        skip_channels = self._parse_skip_channels(parameters_raw)
        delete_after_export = self._coerce_bool_option(
            parameters_raw,
            "delete_after_export",
            default=True,
        )
        include_files = self._coerce_bool_option(
            parameters_raw,
            "include_files",
            default=True,
        )
        include_threads = self._coerce_bool_option(
            parameters_raw,
            "include_threads",
            default=True,
        )
        dry_run = self._coerce_bool_option(
            parameters_raw,
            "dry_run",
            default=False,
        )
        notes = self._parse_notes(parameters_raw)
        return SlackDumpParameters(
            slack_token=token,
            channels=channels,
            archive_root=archive_root,
            delete_after_export=delete_after_export,
            include_files=include_files,
            include_threads=include_threads,
            dry_run=dry_run,
            skip_channels=skip_channels,
            notes=notes,
        )

    @staticmethod
    def _extract_parameters(payload: Mapping[str, object]) -> Mapping[str, object]:
        parameters_obj = payload.get("parameters")
        if isinstance(parameters_obj, Mapping):
            return parameters_obj
        message = "Payload must include a 'parameters' mapping"
        raise RuntimeError(message)

    def _resolve_token(self, parameters_raw: Mapping[str, object]) -> str:
        token_candidate = parameters_raw.get("slack_token")
        token = (
            token_candidate.strip()
            if isinstance(token_candidate, str) and token_candidate.strip()
            else None
        )
        if token and is_valid_slack_access_token(token):
            return token
        env_value = os.getenv("SLACK_TOKEN")
        env_token = (
            env_value.strip()
            if isinstance(env_value, str) and env_value.strip()
            else None
        )
        if env_token and is_valid_slack_access_token(env_token):
            return env_token
        persisted_token, _ = self._persistent_token_resolver()
        if persisted_token and is_valid_slack_access_token(persisted_token):
            return persisted_token
        message = (
            "Slack token not provided in payload or SLACK_TOKEN environment variable"
        )
        raise RuntimeError(message)

    @staticmethod
    def _parse_archive_root(parameters_raw: Mapping[str, object]) -> Path:
        archive_root_raw = parameters_raw.get("archive_root")
        if not isinstance(archive_root_raw, str) or not archive_root_raw.strip():
            message = "archive_root must be a non-empty string path"
            raise RuntimeError(message)
        return Path(archive_root_raw).expanduser().resolve()

    @staticmethod
    def _parse_channels(
        parameters_raw: Mapping[str, object]
    ) -> list[str | Mapping[str, object]]:
        channels_raw = parameters_raw.get("channels")
        if (
            not isinstance(channels_raw, Sequence)
            or isinstance(channels_raw, (str, bytes))
            or not channels_raw
        ):
            message = "channels must be a non-empty array"
            raise RuntimeError(message)
        channels: list[str | Mapping[str, object]] = []
        for item in channels_raw:
            if (isinstance(item, str) and item) or isinstance(item, Mapping):
                channels.append(item)
            else:
                message = "channels entries must be strings or objects with id/name"
                raise RuntimeError(message)
        return channels

    @staticmethod
    def _parse_skip_channels(parameters_raw: Mapping[str, object]) -> set[str]:
        skip_raw = parameters_raw.get("skip_channels")
        if not isinstance(skip_raw, Sequence) or isinstance(skip_raw, (str, bytes)):
            return set()
        return {item for item in skip_raw if isinstance(item, str) and item}

    @staticmethod
    def _coerce_bool_option(
        parameters_raw: Mapping[str, object],
        key: str,
        *,
        default: bool,
    ) -> bool:
        value = parameters_raw.get(key)
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalised = value.strip().lower()
            if normalised in {"true", "1", "yes"}:
                return True
            if normalised in {"false", "0", "no"}:
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        return bool(value)

    @staticmethod
    def _parse_notes(parameters_raw: Mapping[str, object]) -> list[str]:
        notes_raw = parameters_raw.get("notes")
        if not isinstance(notes_raw, Sequence) or isinstance(notes_raw, (str, bytes)):
            return []
        return [note for note in notes_raw if isinstance(note, str)]

    def _resolve_export_root(self, archive_root: Path) -> Path:
        if not archive_root.exists():
            message = f"Archive root does not exist: {archive_root}"
            raise FileNotFoundError(message)
        subdirectories = [item for item in archive_root.iterdir() if item.is_dir()]
        if not subdirectories:
            message = (
                f"Archive root {archive_root} has no subdirectories to target"
            )
            raise FileNotFoundError(message)
        return max(subdirectories, key=lambda item: item.stat().st_mtime)

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
            message = "Channel mapping must provide 'id' or 'name'"
            raise RuntimeError(message)
        if isinstance(channel_spec, str) and channel_spec:
            return channel_spec, channel_spec.lstrip("#")
        message = "Channel specification must be a non-empty string or mapping"
        raise RuntimeError(message)

    @staticmethod
    def _serialise_message(record: SlackMessageRecord) -> JSONObject:
        data: JSONObject = {
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
            data["replies"] = [cast("JSONValue", reply) for reply in record.replies]
        return data


def _load_json_source(path: str | None) -> Mapping[str, object]:
    payload_raw: object
    if path is None or path == "-":
        payload_raw = json.load(sys.stdin)
    else:
        payload_path = Path(path)
        with payload_path.open(encoding="utf-8") as handle:
            payload_raw = json.load(handle)
    payload = _maybe_json_object(payload_raw)
    if payload is None:
        message = "Input payload must be a JSON object"
        raise TypeError(message)
    return payload


def _dump_json(output: Mapping[str, object]) -> None:
    json.dump(output, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


def _write_json_to_path(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


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
        prompt = (
            "Archive captured. Delete Slack messages and files after export?"
            " [y/N]: "
        )
        response = input(prompt).strip().lower()
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
        with suppress(Exception):
            validate_payload(error_payload, ERROR_SCHEMA)
        if args.output:
            _write_json_to_path(Path(args.output), error_payload)
        else:
            _dump_json(error_payload)
        return 1

    if args.output:
        _write_json_to_path(Path(args.output), output)
    else:
        _dump_json(output)
    return 0


if __name__ == "__main__":
    sys.exit(_run_cli())
