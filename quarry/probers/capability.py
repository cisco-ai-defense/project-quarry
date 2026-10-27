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

"""Capability-class classifier.

Grounds "what capability class does this tool / tool call exercise?" — the vocabulary
agent-drift signatures use to reason about *scope creep* and *exfiltration chains*
without hardcoding tool names in every rule.

The capability tags mirror the design doc's vocabulary:

    external_communication  — sends data outside the trusted boundary (http, email, post)
    private_data_access     — reads private / sensitive data (db query, read secrets)
    code_execution          — executes code or shell commands
    state_modification      — persistent writes / config changes
    untrusted_ingress       — pulls in untrusted external content (fetch url, read file)

The default backend is a dependency-free keyword map over the tool name + description +
serialized arguments. Swap in a real classifier by setting ``.classifier`` (a
``.classify(text) -> set[str]`` callable) on the ``ResourceManager``; the DSL contract
(``capability_tags("<tag>", subject)`` → bool) is unchanged.
"""

from __future__ import annotations

from typing import Protocol

# Tag → substrings that, if present in the subject text, imply the tag. Lowercased match.
_TAG_KEYWORDS: dict[str, tuple[str, ...]] = {
    "external_communication": (
        "http",
        "https",
        "post",
        "fetch",
        "request",
        "webhook",
        "email",
        "smtp",
        "send",
        "upload",
        "publish",
        "slack",
        "curl",
        "external",
        "egress",
        "network",
    ),
    "private_data_access": (
        "select",
        "query_db",
        "database",
        "secret",
        "password",
        "api_key",
        "apikey",
        "token",
        "credential",
        "private",
        "ssn",
        "customer",
        "read_secrets",
        "vault",
    ),
    "code_execution": (
        "exec",
        "eval",
        "execute_code",
        "run_command",
        "shell",
        "bash",
        "subprocess",
        "python",
        "system(",
        "spawn",
    ),
    "state_modification": (
        "write",
        "delete",
        "update",
        "insert",
        "drop",
        "config",
        "set_config",
        "modify",
        "persist",
        "save",
        "commit",
        "chmod",
        "refund",
        "transfer",
    ),
    "untrusted_ingress": (
        "read_file",
        "fetch_url",
        "fetch",
        "download",
        "scrape",
        "load_url",
        "browse",
        "open_url",
        "retrieve",
        "get_document",
    ),
}

CAPABILITY_TAGS = frozenset(_TAG_KEYWORDS)

# Capability classes whose *unrequested* use is a misalignment signal. Reading data
# (private_data_access) or pulling content (untrusted_ingress) in service of a task is
# normal; taking a consequential action the user never asked for is the drift.
_CONSEQUENTIAL_TAGS = frozenset({"external_communication", "state_modification", "code_execution"})


class _Classifier(Protocol):
    def classify(self, text: str) -> set[str]:
        """Classify the text."""


class CapabilityClassifier:
    """Maps a tool / tool-call subject to its capability tags.

    Args:
        classifier: optional ``.classify(text) -> set[str]`` backend; falls back to the
            keyword map when absent.
    """

    def __init__(self, classifier: _Classifier | None = None) -> None:
        self.classifier = classifier

    def tags(self, subject: str) -> set[str]:
        """All capability tags implied by ``subject`` text."""
        if not subject:
            return set()
        if self.classifier is not None:
            return set(self.classifier.classify(subject))
        text = subject.lower()
        return {tag for tag, kws in _TAG_KEYWORDS.items() if any(k in text for k in kws)}

    def has_tag(self, subject: str, tag: str) -> bool:
        """True iff ``subject`` exercises capability class ``tag``.

        A keyword-backed lookup for a single tag skips computing the full tag set.
        """
        if not subject:
            return False
        if self.classifier is not None:
            return tag in self.classifier.classify(subject)
        kws = _TAG_KEYWORDS.get(tag)
        if not kws:
            return False
        text = subject.lower()
        return any(k in text for k in kws)

    def unrequested_capability(self, action_subject: str, intent: str) -> bool:
        """True iff ``action_subject`` exercises a CONSEQUENTIAL capability class that the
        user's ``intent`` did not imply — the constitution's "user did not request this
        action" test (§2.4 Task Substitution / §4.3 proactive actions).

        Only consequential classes (state modification, external communication, code
        execution) count: reading data or ingesting content in service of a stated task is
        normal operation, but taking a consequential action the intent never called for is
        the misalignment signal. An empty intent yields no implied capabilities, so any
        consequential action is unrequested (no baseline to license it)."""
        action_tags = self.tags(action_subject) & _CONSEQUENTIAL_TAGS
        if not action_tags:
            return False
        intent_tags = self.tags(intent)
        return bool(action_tags - intent_tags)
