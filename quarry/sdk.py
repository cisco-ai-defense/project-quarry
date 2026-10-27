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

"""Wrapper for using Quarry in Python applications."""

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Any

from quarry.config import QuarryConfig
from quarry.scanner import DSLScanner
from quarry.types.exceptions import ConversationValidationError
from quarry.types.rules import ScanResult

__all__ = ["QuarryGuard"]


class QuarryGuard:
    """Simple API for using Quarry in Python applications."""

    def __init__(self, config_path: str | Path, **kwargs):
        """
        Initialize Quarry guard with configuration file.

        Args:
            config_path: Path to YAML configuration file
            **kwargs: Additional arguments (verbose, etc.)

        Example:
            guard = QuarryGuard("config.yaml")
        """
        self.config = QuarryConfig.from_yaml(Path(config_path))
        # A guard reports every timeout as IncompleteScanError with the partial result, so the
        # low-level re-raise that fail_on_timeout selects is turned off for its scanner.
        limits = kwargs.pop("limits", None) or self.config.get_safety_limits()
        self.scanner = DSLScanner(
            config=self.config, limits=replace(limits, fail_on_timeout=False), **kwargs
        )

        if not self.scanner.loaded_rules:
            raise ValueError("No rules loaded")

    def scan(self, conversation: dict[str, Any]) -> list[dict[str, Any]]:
        """Synchronously scan a conversation and return authoritative matches.

        Fails closed: an invalid conversation or an incomplete scan (a rule that errored,
        timed out, or ran on truncated input) raises via ``raise_for_status`` rather than
        returning a partial match list a caller can't distinguish from a clean pass."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "QuarryGuard.scan() cannot run inside an active event loop; "
                "await QuarryGuard.scan_async() instead"
            )

        self._validate_conversation_type(conversation)
        result = self.scanner.scan_data(conversation)
        return self._matches_from_complete_result(result)

    async def scan_async(self, conversation: dict[str, Any]) -> list[dict[str, Any]]:
        """Asynchronously scan a conversation and return authoritative matches."""
        self._validate_conversation_type(conversation)
        result = await self.scanner.scan_data_async(conversation)
        return self._matches_from_complete_result(result)

    @staticmethod
    def _validate_conversation_type(conversation: Any) -> None:
        if not isinstance(conversation, dict):
            raise ConversationValidationError(
                f"Expected conversation dictionary, got {type(conversation).__name__}"
            )

    def _matches_from_complete_result(self, result: ScanResult) -> list[dict[str, Any]]:
        result.raise_for_status()
        matches = []
        for rule_name in result.rule_matches:
            rule = self.scanner.loaded_rules.get(rule_name)
            if rule:
                match_data = {
                    "rule": rule_name,
                    "metadata": dict(rule.metadata),
                    "source_file": str(rule.source_file) if rule.source_file else None,
                    "timing": (result.rule_timings.get(rule_name, 0.0) * 1000)
                    if result.rule_timings
                    else None,
                }
                output = result.outputs.get(rule_name)
                if output is not None:
                    match_data["output"] = output.to_dict()
                matches.append(match_data)
        return matches
