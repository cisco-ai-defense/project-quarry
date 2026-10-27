#!/usr/bin/env python3
# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Resolve local file markers before Codex or Claude Code calls a Quarry MCP tool."""

from __future__ import annotations

import json
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, cast

RULE_FILE_PREFIX = "quarry+file://"
CONVERSATION_FILE_KEY = "file"
CONVERSATION_BATCH_FILE_KEY = "batch_file"

# Keep these values aligned with the public plugin and default conversation limits. The hook
# applies them before it materializes file contents or sends rewritten arguments to the MCP host.
MAX_RULE_TEXT_CHARS = 100_000
MAX_RULE_FILE_BYTES = MAX_RULE_TEXT_CHARS * 4
MAX_SCAN_BATCH = 1_000
MAX_CONVERSATION_BYTES = 50_000_000
MAX_FILE_REFERENCE_CHARS = 500

_QUARRY_TOOL = re.compile(r"^mcp__(?:quarry|plugin_quarry_quarry)__(quarry_compile|quarry_scan)$")
_LOCAL_FILE_CONTEXT = """Local Quarry file inputs are available through trusted host hooks.
For quarry_compile or quarry_scan, use quarry+file://path/to/rule.rule as rule_text.
Use {"file":"path/to/conversation.json"} for one conversation or each conversations item.
Use one {"batch_file":"path/to/batch.json"} item for a JSON array.
Use relative workspace paths only. Do not read these files into model context first."""


class HydrationError(ValueError):
    """A local file marker cannot be resolved safely."""


@dataclass(frozen=True)
class _FileRequest:
    purpose: str
    reference: str


@dataclass(frozen=True)
class _ResolvedFile:
    purpose: str
    reference: str
    path: Path
    size: int
    device: int
    inode: int
    modified_ns: int

    @property
    def max_bytes(self) -> int:
        return MAX_RULE_FILE_BYTES if self.purpose == "rule" else MAX_CONVERSATION_BYTES


def _tool_kind(tool_name: object) -> str | None:
    if not isinstance(tool_name, str):
        return None
    match = _QUARRY_TOOL.search(tool_name)
    return match.group(1) if match else None


def _marker_reference(value: object, key: str, owner: str) -> str | None:
    if not isinstance(value, dict) or key not in value:
        return None
    fields: dict[object, object] = dict(value)
    if set(fields) != {key}:
        raise HydrationError(f"{owner} file marker must contain only {key!r}")
    reference = fields[key]
    if not isinstance(reference, str):
        raise HydrationError(f"{owner} file marker must contain a string path")
    return reference


def _collect_requests(tool_kind: str, arguments: dict[str, Any]) -> tuple[list[_FileRequest], bool]:
    requests: list[_FileRequest] = []
    found_marker = False

    rule_text = arguments.get("rule_text")
    if isinstance(rule_text, str) and rule_text.startswith(RULE_FILE_PREFIX):
        requests.append(_FileRequest("rule", rule_text.removeprefix(RULE_FILE_PREFIX)))
        found_marker = True

    if tool_kind != "quarry_scan":
        return requests, found_marker

    conversation_reference = _marker_reference(
        arguments.get("conversation"), CONVERSATION_FILE_KEY, "conversation"
    )
    if conversation_reference is not None:
        requests.append(_FileRequest("conversation", conversation_reference))
        found_marker = True

    conversations = arguments.get("conversations")
    if not isinstance(conversations, list):
        return requests, found_marker

    batch_markers = [
        (index, reference)
        for index, item in enumerate(conversations)
        if (
            reference := _marker_reference(
                item,
                CONVERSATION_BATCH_FILE_KEY,
                f"conversations[{index}]",
            )
        )
        is not None
    ]
    if batch_markers:
        if len(conversations) != 1 or len(batch_markers) != 1:
            raise HydrationError(
                f"{CONVERSATION_BATCH_FILE_KEY!r} must be the only item in conversations"
            )
        requests.append(_FileRequest("batch", batch_markers[0][1]))
        return requests, True

    for index, item in enumerate(conversations):
        reference = _marker_reference(
            item,
            CONVERSATION_FILE_KEY,
            f"conversations[{index}]",
        )
        if reference is not None:
            requests.append(_FileRequest("conversation", reference))
            found_marker = True
    return requests, found_marker


def _workspace_root(event: dict[str, Any]) -> Path:
    cwd = event.get("cwd")
    if not isinstance(cwd, str) or not cwd:
        raise HydrationError("the hook request has no working directory")
    try:
        root = Path(cwd).resolve(strict=True)
    except OSError as exc:
        raise HydrationError("the hook working directory is not available") from exc
    if not root.is_dir():
        raise HydrationError("the hook working directory is not a directory")
    return root


def _resolve_file(root: Path, request: _FileRequest) -> _ResolvedFile:
    reference = request.reference
    owner = f"{request.purpose} file"
    if not reference or reference != reference.strip():
        raise HydrationError(f"{owner} path must be a non-empty trimmed string")
    if len(reference) > MAX_FILE_REFERENCE_CHARS:
        raise HydrationError(f"{owner} path must not exceed {MAX_FILE_REFERENCE_CHARS} characters")
    if "\x00" in reference:
        raise HydrationError(f"{owner} path must not contain a NUL byte")
    if "\\" in reference:
        raise HydrationError(f"{owner} path must use '/' as the separator")
    if reference.startswith("~"):
        raise HydrationError(f"{owner} path must not start with '~'")

    portable = PurePosixPath(reference)
    if portable.is_absolute() or PureWindowsPath(reference).drive:
        raise HydrationError(f"{owner} path must be relative to the workspace")
    if ".." in portable.parts:
        raise HydrationError(f"{owner} path must not contain '..'")
    expected_suffix = ".rule" if request.purpose == "rule" else ".json"
    if portable.suffix.lower() != expected_suffix:
        raise HydrationError(f"{owner} path must end with {expected_suffix!r}")

    try:
        candidate = (root / Path(*portable.parts)).resolve(strict=True)
        candidate.relative_to(root)
    except (OSError, ValueError) as exc:
        raise HydrationError(f"{owner} path must resolve inside the workspace") from exc
    if not candidate.is_file():
        raise HydrationError(f"{owner} path must identify a regular file")

    try:
        metadata = candidate.stat()
    except OSError as exc:
        raise HydrationError(f"{owner} cannot be measured") from exc
    size = metadata.st_size
    max_bytes = MAX_RULE_FILE_BYTES if request.purpose == "rule" else MAX_CONVERSATION_BYTES
    if size > max_bytes:
        raise HydrationError(f"{owner} exceeds its {max_bytes}-byte limit")
    return _ResolvedFile(
        request.purpose,
        reference,
        candidate,
        size,
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mtime_ns,
    )


def _prepare_files(
    root: Path, requests: list[_FileRequest]
) -> dict[tuple[str, str], _ResolvedFile]:
    resolved: dict[tuple[str, str], _ResolvedFile] = {}
    total_conversation_bytes = 0
    for request in requests:
        item = _resolve_file(root, request)
        resolved[(request.purpose, request.reference)] = item
        if request.purpose in {"conversation", "batch"}:
            total_conversation_bytes += item.size
    if total_conversation_bytes > MAX_CONVERSATION_BYTES:
        raise HydrationError(
            f"conversation files exceed the {MAX_CONVERSATION_BYTES}-byte request limit in total"
        )
    return resolved


def _read_utf8(item: _ResolvedFile) -> str:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(item.path, flags)
        with os.fdopen(descriptor, "rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise HydrationError(f"{item.purpose} file must remain a regular file")
            if (before.st_dev, before.st_ino) != (item.device, item.inode):
                raise HydrationError(f"{item.purpose} file changed after validation")
            if before.st_size != item.size or before.st_mtime_ns != item.modified_ns:
                raise HydrationError(f"{item.purpose} file changed after validation")
            if before.st_size > item.max_bytes:
                raise HydrationError(f"{item.purpose} file exceeds its {item.max_bytes}-byte limit")
            data = handle.read(item.max_bytes + 1)
            after = os.fstat(handle.fileno())
            if after.st_size != before.st_size or after.st_mtime_ns != before.st_mtime_ns:
                raise HydrationError(f"{item.purpose} file changed while it was read")
    except HydrationError:
        raise
    except OSError as exc:
        raise HydrationError(f"{item.purpose} file cannot be read") from exc
    if len(data) > item.max_bytes:
        raise HydrationError(f"{item.purpose} file exceeds its {item.max_bytes}-byte limit")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HydrationError(f"{item.purpose} file must be valid UTF-8") from exc


def _read_rule(item: _ResolvedFile) -> str:
    text = _read_utf8(item)
    if not text.strip():
        raise HydrationError("rule file must not be empty")
    if len(text) > MAX_RULE_TEXT_CHARS:
        raise HydrationError(f"rule file must not exceed {MAX_RULE_TEXT_CHARS} characters")
    return text


def _read_json(item: _ResolvedFile) -> object:
    text = _read_utf8(item)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise HydrationError(
            f"{item.purpose} file contains invalid JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc


def _read_conversation(item: _ResolvedFile) -> dict[str, Any]:
    value = _read_json(item)
    if not isinstance(value, dict):
        raise HydrationError("conversation file must contain one JSON object")
    return cast(dict[str, Any], value)


def _read_batch(item: _ResolvedFile) -> list[dict[str, Any]]:
    value = _read_json(item)
    if not isinstance(value, list):
        raise HydrationError("batch file must contain one JSON array")
    if not 1 <= len(value) <= MAX_SCAN_BATCH:
        raise HydrationError(
            f"batch file must contain between 1 and {MAX_SCAN_BATCH} conversations"
        )
    if any(not isinstance(conversation, dict) for conversation in value):
        raise HydrationError("every batch file item must be a JSON object")
    return cast(list[dict[str, Any]], value)


def hydrate_event(event: dict[str, Any]) -> dict[str, Any] | None:
    """Return a PreToolUse decision when the event contains local Quarry file markers."""

    hook_event = event.get("hook_event_name")
    if hook_event == "SessionStart":
        return {
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": _LOCAL_FILE_CONTEXT,
            }
        }
    if hook_event != "PreToolUse":
        return None
    tool_kind = _tool_kind(event.get("tool_name"))
    if tool_kind is None:
        return None
    arguments = event.get("tool_input")
    if not isinstance(arguments, dict):
        raise HydrationError("the Quarry tool input must be a JSON object")

    requests, found_marker = _collect_requests(tool_kind, arguments)
    if not found_marker:
        return None
    root = _workspace_root(event)
    files = _prepare_files(root, requests)
    updated = dict(arguments)

    rule_text = updated.get("rule_text")
    if isinstance(rule_text, str) and rule_text.startswith(RULE_FILE_PREFIX):
        reference = rule_text.removeprefix(RULE_FILE_PREFIX)
        updated["rule_text"] = _read_rule(files[("rule", reference)])

    if tool_kind == "quarry_scan":
        conversation_reference = _marker_reference(
            updated.get("conversation"), CONVERSATION_FILE_KEY, "conversation"
        )
        if conversation_reference is not None:
            updated["conversation"] = _read_conversation(
                files[("conversation", conversation_reference)]
            )

        conversations = updated.get("conversations")
        if isinstance(conversations, list):
            batch_reference = (
                _marker_reference(
                    conversations[0],
                    CONVERSATION_BATCH_FILE_KEY,
                    "conversations[0]",
                )
                if len(conversations) == 1
                else None
            )
            if batch_reference is not None:
                updated["conversations"] = _read_batch(files[("batch", batch_reference)])
            else:
                hydrated = None
                for index, conversation in enumerate(conversations):
                    reference = _marker_reference(
                        conversation,
                        CONVERSATION_FILE_KEY,
                        f"conversations[{index}]",
                    )
                    if reference is not None:
                        if hydrated is None:
                            hydrated = list(conversations)
                        hydrated[index] = _read_conversation(files[("conversation", reference)])
                if hydrated is not None:
                    updated["conversations"] = hydrated

    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "updatedInput": updated,
        }
    }


def _deny(message: str) -> dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": f"Quarry local file input failed: {message}",
        }
    }


def main() -> int:
    try:
        event = json.load(sys.stdin)
        if not isinstance(event, dict):
            raise HydrationError("the hook request must be a JSON object")
        result = hydrate_event(event)
    except HydrationError as exc:
        result = _deny(str(exc))
    except Exception:
        result = _deny("unexpected hook failure")

    if result is not None:
        json.dump(result, sys.stdout, ensure_ascii=False, separators=(",", ":"))
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
