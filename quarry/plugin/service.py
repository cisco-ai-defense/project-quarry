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

"""Host-neutral implementation of the Quarry agent tools.

This module contains no MCP dependencies. Hosts can call the service
directly or expose it through :mod:`quarry.plugin.server`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import secrets
from collections import defaultdict
from collections.abc import Mapping
from contextlib import suppress
from copy import deepcopy
from dataclasses import replace
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import yaml

from quarry import __version__ as QUARRY_VERSION
from quarry.cache.async_cache import result_cache
from quarry.config import QuarryConfig
from quarry.core.logging import get_app_logger
from quarry.decision_report import path_report, report_tree, scanner_digests, tree_report
from quarry.integrations.model_inference import ModelInferenceBackend
from quarry.integrations.rate_limiter import LLMRateLimiter
from quarry.integrations.resources import ResourcePreparationError
from quarry.language import get_language_spec
from quarry.plugin.visualization import (
    build_batch_runtime_attribution,
    build_runtime_attribution,
    build_visualization_bundle,
)
from quarry.policy import bind_prepared_policy
from quarry.prepared import PreparedRuleCache
from quarry.safety.quality import quality_warnings
from quarry.safety.validator import Complexity, RuleValidator
from quarry.scanner import DSLScanner
from quarry.types.exceptions import (
    ConfigurationError,
    ConversationValidationError,
    IncompleteScanError,
    QuarryError,
    ResourceError,
    RuleParseError,
    RuleValidationError,
)
from quarry.types.parameters import ParameterValueError

_logger = get_app_logger()

PLUGIN_VERSION = "0.2.0"
MAX_REFERENCE_QUERY_CHARS = 4_096
MAX_REFERENCE_IDS = 50
MAX_REFERENCE_ID_CHARS = 200
MAX_RULE_TEXT_CHARS = 100_000
MAX_PARAMETER_TEXT_CHARS = 100_000
MAX_SCAN_BATCH = 1_000
MAX_FLOW_PATHS = 100_000
MAX_FLOW_LABEL_CHARS = 200
MAX_PREPARED_RULES = 64
_COMPLEXITY_RANK = {level: rank for rank, level in enumerate(Complexity)}
ReferenceAction = Literal["index", "search", "get", "rejected"]
ReferenceKind = Literal["function", "compiler_form", "surface", "construct", "token", "example"]
ReferenceVerbosity = Literal["compact", "full"]

# Keys the catalog carries for Quarry's own compiler and code generators. A rule author cannot
# write them and an agent cannot act on them, so the reference omits them rather than spending
# response budget on the plugin's own bookkeeping. Dropped at the top level of a record only:
# ``resource_group`` is bookkeeping there but is the answer under ``availability``.
_INTERNAL_RECORD_KEYS = frozenset(
    {
        "accessor_filter",
        "accessor_surface",
        "compiler_tags",
        "handler",
        "implicit_receiver_count",
        "resource_group",
        "source",
    }
)

# Every flag here names a restriction or a precondition, so an absent key reads as "unrestricted"
# and says exactly what ``false`` said. A flag that instead asserts a property of the thing is
# never listed, however lopsided its values are: reading its absence as ``false`` would read a
# claim the catalog never made. ``model_input`` is the case that separates the two -- a surface
# that is not fed to the model is a positive, security-relevant fact about that surface, not the
# absence of a limit on it. ``required`` on a parameter and ``expected_match`` on an example
# check are likewise always spelled out.
_OMITTED_WHEN_FALSE = frozenset(
    {
        "deprecated",
        "field_only",
        "literal_only",
        "requires_backend",
        "reserved_for_variables",
        "variadic",
    }
)

# A value under one of these keys is data an example feeds to the scanner, not a catalog
# record. It passes through exactly as the example holds it, because an empty field there is
# an input the test actually ran, and pruning it would describe a different conversation.
_OPAQUE_RECORD_KEYS = frozenset({"input"})

# The empty-collection allowlist. ``None`` and ``""`` are the absence of a value for every
# key, so a missing key says the same and they are always dropped. An empty *collection* is
# not always an absence: for many fields ``[]`` is a reported fact the catalog states on
# purpose. ``allowed_uses: []`` means the item is allowed in no expression position -- the
# compiler checks membership in this field -- and dropping it would leave an agent unable to
# tell "allowed nowhere" from "not reported". So an empty list or dict is kept unless its key
# is named here, an explicit allowlist of fields whose empty collection is genuinely
# redundant. ``allowed_values`` is the one such field: an empty enumeration on a parameter
# accepts any value of its accepted types, which is exactly what an absent key already says.
_OMITTED_WHEN_EMPTY = frozenset({"allowed_values"})

_WORD = re.compile(r"[a-z0-9]+")
_RULE_STRING_OR_COMMENT = re.compile(r'"(?:\\.|[^"\\])*"|//[^\r\n]*')
_RULE_CALL = re.compile(r"(?<![A-Za-z0-9_.])([A-Za-z_][A-Za-z0-9_.]*)\s*\(")
_RULE_POSTFIX_CALL = re.compile(r"\.\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_CALLABLE_KINDS = frozenset({"function", "compiler_form"})
_STOP_WORDS = frozenset({"a", "an", "and", "by", "for", "in", "is", "of", "or", "the", "to"})
_TOKEN_ALIASES = {
    "called": "call",
    "calls": "call",
    "events": "event",
    "followed": "after",
    "following": "after",
    "follows": "after",
    "immediately": "right",
    "invoked": "call",
    "invokes": "call",
    "preceded": "before",
    "precedes": "before",
    "responses": "response",
    "three": "3",
    "tools": "tool",
}
# Compound identifiers that carry no separator are split so their meaningful part is
# searchable. `right_after` already splits on the underscore; `nocase`/`fullword` do not,
# so without this a search for "case insensitive" would never reach `nocase`. Only the
# discriminating segment is indexed -- adding the generic prefixes "no"/"full" would let
# unrelated plain-language queries mis-rank these modifier records. Applied to indexed
# catalog text only, never to a query; see `_indexed_tokens`.
_COMPOUND_TOKENS = {
    "nocase": ("case",),
    "fullword": ("word",),
}
_SEARCH_FIELDS = (
    ("id", 70),
    ("name", 70),
    ("spelling", 70),
    ("signature", 45),
    ("syntax", 45),
    ("summary", 30),
    ("description", 20),
    ("parameters", 12),
    # Whole example rules are useful recall evidence, but include many unrelated words.
    ("rule_text", 12),
    ("example_text", 8),
)


def _receiver_before_dot(code: str, dot: int) -> str | None:
    """Find the immediate receiver of a method call in string-free rule text."""

    def receiver_ending_at(end: int) -> tuple[str, int] | None:
        index = end - 1
        while index >= 0 and code[index].isspace():
            index -= 1
        if index < 0:
            return None
        if code[index] == ")":
            closing = index
            depth = 1
            index -= 1
            while index >= 0 and depth:
                if code[index] == ")":
                    depth += 1
                elif code[index] == "(":
                    depth -= 1
                index -= 1
            if depth:
                return None
            opening = index + 1
            call = receiver_ending_at(opening)
            if call is not None and call[1] < opening:
                return call
            # A parenthesized receiver is valid only when the entire group is that
            # receiver, not when a call is merely the tail of a larger expression.
            group = receiver_ending_at(closing)
            if group is not None and not code[opening + 1 : group[1]].strip():
                return group[0], opening
            return None
        end_of_name = index + 1
        while index >= 0 and (code[index].isalnum() or code[index] == "_"):
            index -= 1
        return (code[index + 1 : end_of_name], index + 1) if index + 1 < end_of_name else None

    receiver = receiver_ending_at(dot)
    return receiver[0] if receiver is not None else None


def _rule_call_names(rule_text: str, callable_names: set[str]) -> set[str]:
    """Map direct and receiver-style calls in an example to catalog callable names."""
    code = _RULE_STRING_OR_COMMENT.sub(" ", rule_text)
    names = set()
    for match in _RULE_CALL.finditer(code):
        name = match.group(1)
        if name in callable_names:
            names.add(name)
        elif "." in name and (method := name.rsplit(".", 1)[1]) in callable_names:
            names.add(method)
    for match in _RULE_POSTFIX_CALL.finditer(code):
        method = match.group(1)
        receiver = _receiver_before_dot(code, match.start())
        qualified = f"{receiver}.{method}" if receiver else ""
        if qualified in callable_names:
            names.add(qualified)
        elif method in callable_names:
            names.add(method)
    return names


def _prune(value: Any) -> Any:
    """Return ``value`` without the keys that carry nothing the caller can use.

    ``None``, an empty string and an unset flag all say "not applicable here", which a
    missing key says just as well. An empty collection is kept unless its key is in
    ``_OMITTED_WHEN_EMPTY``, because for most fields an empty list or dict is a reported
    fact rather than an absence; see that set for the distinction. Pruning only ever removes
    keys, so every value that does arrive is the value the catalog holds, unrounded and
    untruncated.
    """

    if isinstance(value, dict):
        pruned = {}
        for key, item in value.items():
            if key in _OPAQUE_RECORD_KEYS:
                pruned[key] = deepcopy(item)
                continue
            if item is None or item == "":
                continue
            if isinstance(item, (list, dict)) and not item and key in _OMITTED_WHEN_EMPTY:
                continue
            if item is False and key in _OMITTED_WHEN_FALSE:
                continue
            pruned[key] = _prune(item)
        return pruned
    if isinstance(value, list):
        return [_prune(item) for item in value]
    return value


class PluginInputError(ValueError):
    """A caller supplied an invalid tool request."""


def load_runtime_config(path: str | Path | None = None) -> QuarryConfig | None:
    """Discover and load trusted server configuration without accepting rule paths.

    The MCP tools always receive the rule text in the request. Keeping ``rules`` out of
    this file prevents a server operator's unrelated rules from entering a tool call.

    An explicit path overrides the user-level ``~/.quarry/plugin.yaml`` file. When
    neither exists, the plugin starts without model-backed capabilities so local rules
    continue to work without setup.
    """

    if path is None:
        config_path = Path.home() / ".quarry" / "plugin.yaml"
        if not config_path.exists():
            return None
    else:
        if isinstance(path, str) and not path.strip():
            raise ValueError("Quarry plugin config path must not be empty")
        config_path = Path(path).expanduser()

    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Quarry plugin config not found: {config_path}")
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Quarry plugin config must be a mapping: {config_path}")
    if "rules" in document:
        raise ValueError("Quarry plugin config must not contain 'rules'; tools supply rule_text")
    return QuarryConfig.from_mapping(document, config_path.parent, require_rules=False)


def _brief_report(report: dict[str, Any], *, visualize: bool) -> dict[str, Any]:
    """A compile report as ``compile`` and ``scan`` return it: without the decision tree,
    which ``decision_tree`` returns, and without the static source graph unless the
    caller visualizes, so an agent compiling or scanning a rule reads only what it needs."""

    omitted = {"decision_tree"} if visualize else {"decision_tree", "visualization"}
    return {key: value for key, value in report.items() if key not in omitted}


class QuarryPlugin:
    """Implement the five read-only Quarry tools against one installed Quarry version."""

    def __init__(self, runtime_config: QuarryConfig | None = None):
        if runtime_config is not None:
            # Report at startup each configured model that cannot get credentials, so the host
            # learns it before the first scan. Only a rule that uses that model fails to load,
            # and the other tools keep working.
            for model_id in sorted(runtime_config.models):
                try:
                    backend = ModelInferenceBackend(runtime_config, models_to_load=[model_id])
                    backend.require_credentials([model_id])
                except ConfigurationError as exc:
                    _logger.warning(f"{exc}. A rule that uses this model will fail to load.")
        self._runtime_config = runtime_config
        # Use one unpredictable key for this server's tree IDs and policy evidence. A scan
        # must name the same tree returned by quarry_decision_tree for this server instance.
        self._evidence_key = secrets.token_bytes(32)
        # Compiled rules by (rule hash, config version). A host scans the same
        # rule every turn; the trusted runtime configuration is fixed for this instance,
        # so the rule text alone decides the result. Filled from worker threads.
        self._prepared = PreparedRuleCache[tuple[dict[str, Any], DSLScanner | None]](
            MAX_PREPARED_RULES
        )
        self._rate_limiter = (
            LLMRateLimiter(runtime_config.llm_rate_limits) if runtime_config is not None else None
        )
        # The server is the process's long-lived host, so its configuration bounds the
        # process's result cache, which every scan of every rule shares.
        if runtime_config is not None:
            result_cache.configure(
                max_entries=runtime_config.model_cache.max_entries,
                ttl_seconds=runtime_config.model_cache.ttl_seconds,
            )
        self._catalog = get_language_spec()
        self._items = {item["id"]: item for item in self._catalog["items"]}
        self._examples = {
            example["id"]: self._materialize_example(example)
            for example in self._catalog["examples"]
        }
        self._items_by_name = {
            item.get("name", item.get("spelling")): item
            for item in self._catalog["items"]
            if item.get("name") or item.get("spelling")
        }

    def reference(
        self,
        *,
        action: ReferenceAction,
        query: str | None = None,
        ids: list[str] | None = None,
        kinds: list[ReferenceKind] | None = None,
        limit: int | None = None,
        offset: int = 0,
        verbosity: ReferenceVerbosity | None = None,
        include_precedence: bool | None = None,
    ) -> dict[str, Any]:
        """Browse, search, or retrieve records from the installed language catalog.

        ``rejected`` reads the separate table of spellings Quarry refuses; it returns that
        bounded table whole and so does not paginate.

        ``limit`` bounds the results of ``index`` and ``search`` and defaults to 10 for them.
        ``get`` and ``rejected`` are each asked for a set the caller already delimited -- an id
        list, or the whole table -- so they take no ``limit`` and reject one that is supplied,
        returning the set and reporting what they did with every id.

        ``verbosity`` selects how much of a ``get`` record is spelled out: ``compact`` prunes
        every empty and null field for its budget, while ``full`` keeps them, along with each
        example's coverage list and conversation, so an absence can be read as one.
        ``include_precedence`` asks for or refuses the operator-precedence table on any action;
        left unset, it arrives with ``index``, the action an author calls to orient themselves.
        """

        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 1 or limit > 50
        ):
            raise PluginInputError("limit must be an integer between 1 and 50")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise PluginInputError("offset must be a non-negative integer")
        if kinds is not None and len(kinds) > 6:
            raise PluginInputError("kinds accepts at most 6 values")
        selected_kinds = set(kinds or [])
        unknown_kinds = selected_kinds - {
            "function",
            "compiler_form",
            "surface",
            "construct",
            "token",
            "example",
        }
        if unknown_kinds:
            raise PluginInputError(f"unknown kinds: {', '.join(sorted(unknown_kinds))}")
        if verbosity is not None and verbosity not in ("compact", "full"):
            raise PluginInputError("verbosity must be 'compact' or 'full'")
        if verbosity is not None and action != "get":
            raise PluginInputError("verbosity applies only to get")
        if include_precedence is not None and not isinstance(include_precedence, bool):
            raise PluginInputError("include_precedence must be true or false")

        response = self._version_envelope()
        response["action"] = action
        response["runtime_capabilities"] = self._runtime_capabilities()
        # The table is the same on every call and is what an author reads once, before writing
        # anything, so it rides along with the browsing action and is asked for explicitly
        # anywhere else. Repeating it on every search and every get charged the caller for it
        # again each time without ever telling them anything new.
        wants_precedence = action == "index" if include_precedence is None else include_precedence
        if wants_precedence:
            response["operator_precedence"] = deepcopy(self._catalog["operator_precedence"])

        if action == "index":
            if query is not None or ids is not None:
                raise PluginInputError(
                    "index accepts only kinds, limit, offset, and include_precedence"
                )
            entries = self._index_entries(selected_kinds)
            page = 10 if limit is None else limit
            next_offset = offset + page
            response.update(
                {
                    "total": len(entries),
                    "offset": offset,
                    "next_offset": next_offset if next_offset < len(entries) else None,
                    "results": entries[offset:next_offset],
                }
            )
            return response

        if action == "search":
            if offset:
                raise PluginInputError("search does not accept offset")
            if not isinstance(query, str) or not query.strip():
                raise PluginInputError("search requires a non-empty query")
            if len(query) > MAX_REFERENCE_QUERY_CHARS:
                raise PluginInputError(
                    f"query must not exceed {MAX_REFERENCE_QUERY_CHARS} characters"
                )
            if ids is not None:
                raise PluginInputError("search does not accept ids")
            all_results = self._search(query, selected_kinds)
            page = 10 if limit is None else limit
            response.update(
                {"query": query, "total": len(all_results), "results": all_results[:page]}
            )
            return response

        if action == "rejected":
            if ids is not None or kinds is not None or offset or limit is not None:
                raise PluginInputError(
                    "rejected accepts only an optional query and include_precedence"
                )
            if query is not None and (not isinstance(query, str) or not query.strip()):
                raise PluginInputError("query must be a non-empty string when supplied")
            if query is not None and len(query) > MAX_REFERENCE_QUERY_CHARS:
                raise PluginInputError(
                    f"query must not exceed {MAX_REFERENCE_QUERY_CHARS} characters"
                )
            # The table is returned whole rather than paginated, and so ignores limit: an
            # author needs the complete list of mistakes to avoid before writing anything,
            # and honouring a small default limit would cut the list short of that purpose.
            # Generation enforces MAX_COMPATIBILITY_RECORDS, so "whole" stays affordable.
            entries = self._rejected_entries(query)
            if query is not None:
                response["query"] = query
            response.update({"total": len(entries), "results": entries})
            return response

        if action == "get":
            if offset:
                raise PluginInputError("get does not accept offset")
            # get takes no limit rather than silently ignoring one: the caller enumerated
            # exactly what it wants in ``ids``, so a supplied limit could only contradict that
            # list. Accepting it silently let a ``limit=1`` request quietly return every record.
            if limit is not None:
                raise PluginInputError("get does not accept limit; it returns every id it is given")
            if query is not None or not ids:
                raise PluginInputError("get requires one or more ids and does not accept query")
            # The request bound is on what the caller submitted, checked before anything reduces
            # the list. Deduplicating first would let an arbitrarily long list of repeats slip
            # past the 50-id limit the schema advertises.
            if len(ids) > MAX_REFERENCE_IDS:
                raise PluginInputError(
                    f"get accepts at most {MAX_REFERENCE_IDS} ids but received {len(ids)}"
                )
            if any(not isinstance(record_id, str) or not record_id for record_id in ids):
                raise PluginInputError("ids must contain non-empty strings")
            if any(len(record_id) > MAX_REFERENCE_ID_CHARS for record_id in ids):
                raise PluginInputError(
                    f"each id must not exceed {MAX_REFERENCE_ID_CHARS} characters"
                )
            # With the request size already bounded, duplicates are removed for lookup and for
            # the response's accounting, so a repeated id yields its record once rather than
            # once per mention.
            unique_ids = list(dict.fromkeys(ids))
            records = []
            missing = []
            filtered = []
            for record_id in unique_ids:
                record = self._full_record(record_id, verbosity or "compact")
                if record is None:
                    missing.append(record_id)
                elif selected_kinds and record["kind"] not in selected_kinds:
                    # An id that exists but is not of a requested kind is reported rather than
                    # dropped. Silence here read as "no such record", which sent an agent back
                    # to search for something it was already holding the id of.
                    filtered.append(record_id)
                else:
                    records.append(record)
            # ``missing_ids`` and ``filtered_ids`` are reported even when empty. Pruning applies to
            # the records inside a response, not to this accounting: an agent that cannot see
            # whether it passed ``kinds`` would have to guess whether an absent key meant "nothing
            # was filtered" or "this build does not report filtering".
            response.update(
                {
                    "verbosity": verbosity or "compact",
                    "results": records,
                    "missing_ids": missing,
                    "filtered_ids": filtered,
                }
            )
            return response

        raise PluginInputError(f"unsupported reference action: {action}")

    def compile(
        self,
        *,
        rule_text: str,
        config_version: int | None = None,
        visualize: bool = False,
    ) -> dict[str, Any]:
        """Compile one complete rule through Quarry's public scanner construction path.

        The report states whether the rule compiles and what it needs, and stays small:
        the decision tree is ``decision_tree``'s, and the static source graph is added
        only with ``visualize``, as for ``scan``.
        """

        report, _ = self._prepare_rule(rule_text, config_version)
        return _brief_report(report, visualize=visualize)

    def decision_tree(self, *, rule_text: str, config_version: int | None = None) -> dict[str, Any]:
        """Compile one rule and return its decision tree.

        The response carries the rule's identity and, for a rule that compiles, its tree:
        its checks, each check's true and false edges, and its root. A rule that does not
        compile returns its diagnostics instead, as ``compile`` does.
        """

        report, _ = self._prepare_rule(rule_text, config_version)
        response = {key: report[key] for key in ("plugin_version", "quarry_version", "rule_hash")}
        response.update({"valid": report["valid"], "diagnostics": report["diagnostics"]})
        if report["valid"]:
            response["decision_tree"] = report["decision_tree"]
        return response

    async def scan(
        self,
        *,
        rule_text: str,
        conversation: dict[str, Any] | None = None,
        conversations: list[dict[str, Any]] | None = None,
        parameters: Mapping[str, object] | None = None,
        explain: bool | Literal["support"] = True,
        visualize: bool = False,
        config_version: int | None = None,
        explore_rate: float = 0.0,
    ) -> dict[str, Any]:
        """Compile one rule once and scan one conversation or an ordered batch.

        A model call this server already made with the same model and input, within
        ``model_cache.ttl_seconds`` of its last use, is not made again, whichever scan made
        it.

        ``visualize`` adds the static source graph once under ``compilation`` and a
        runtime-only ``visualization`` overlay; without it neither is recorded or returned.

        Every complete result of a one-rule text carries its ``decision_path``: the
        conversation's hash, the tree nodes it reached with their outcomes and scores,
        and its leaf. ``explore_rate`` selects that fraction of conversations, by hash,
        whose path also evaluates every node it skipped.
        """

        if (conversation is None) == (conversations is None):
            raise PluginInputError("provide exactly one of conversation or conversations")
        if conversations is not None and not 1 <= len(conversations) <= MAX_SCAN_BATCH:
            raise PluginInputError(
                f"conversations must contain between 1 and {MAX_SCAN_BATCH} items"
            )
        selected = [conversation] if conversation is not None else conversations or []
        if any(not isinstance(item, dict) for item in selected):
            raise PluginInputError("every conversation must be a JSON object")
        if (
            isinstance(explore_rate, bool)
            or not isinstance(explore_rate, int | float)
            or not 0 <= explore_rate <= 1
        ):
            raise PluginInputError("explore_rate must be a number from 0 to 1")

        def fail_before_scan(diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
            response.update(
                {
                    "completion_state": "invalid",
                    "diagnostics": diagnostics,
                    "timings": {},
                }
            )
            if conversations is not None:
                response.update({"results": [], "count": 0})
            return response

        compilation, scanner = await asyncio.to_thread(
            self._prepare_rule,
            rule_text,
            config_version,
        )
        response = self._version_envelope()
        response["rule_hash"] = self._rule_hash(rule_text)
        response["compilation"] = _brief_report(compilation, visualize=visualize)
        static_graphs = compilation.get("visualization") if visualize else None
        if scanner is None:
            return fail_before_scan(compilation["diagnostics"])
        if len(scanner.compiled_rules) != 1:
            error = RuleValidationError(
                "rule_text must define exactly one rule to scan; found "
                + ", ".join(sorted(scanner.compiled_rules))
                + ".",
                {"error_type": "multiple_public_rules"},
            )
            return fail_before_scan([self._exception_diagnostic(error)])

        try:
            supplied = dict(parameters or {})
            parameter_text_chars = sum(
                len(value) for value in supplied.values() if isinstance(value, str)
            )
            if parameter_text_chars > MAX_PARAMETER_TEXT_CHARS:
                raise ParameterValueError(
                    f"parameter text must not exceed {MAX_PARAMETER_TEXT_CHARS} characters in total"
                )
            policy, scan_values = bind_prepared_policy(
                scanner,
                rule_text,
                supplied,
                evidence_key=self._evidence_key,
            )
        except ParameterValueError as exc:
            return fail_before_scan([{"code": "invalid_parameters", "message": str(exc)}])

        try:
            results = await policy.scan_many_async(
                selected,
                scan_values,
                explain=explain,
                record_traces=visualize,
                decision_paths=True,
                explore_rate=float(explore_rate),
            )
        except ParameterValueError as exc:
            return fail_before_scan([{"code": "invalid_parameters", "message": str(exc)}])

        provenance = results[0]
        response.update(
            {
                "policy_digest": policy.policy_digest,
                "parameter_digest": provenance.parameter_digest,
                "parameter_evidence": provenance.parameter_evidence,
            }
        )

        ((rule_name, artifact),) = scanner.compiled_rules.items()
        tree = report_tree(artifact, scanner_digests(scanner, self._evidence_key))
        payloads = []
        for item, result in zip(selected, results, strict=True):
            # ``scan_many_async`` stamps every result, timed-out ones included, with the digest.
            assert result.parameter_digest is not None
            payload = self._scan_payload(scanner, result, static_graphs, node_ids=tree.ids)
            payload["decision_path"] = path_report(
                tree,
                (result.decision_paths or {}).get(rule_name),
                matched=rule_name in result.rule_matches if result.is_complete else None,
                conversation=item,
                parameter_digest=result.parameter_digest,
                explain=bool(explain),
                explore_rate=float(explore_rate),
            )
            payloads.append(payload)
        if conversation is not None:
            response.update(payloads[0])
        else:
            states = {result.completion_state for result in results}
            response["completion_state"] = (
                "complete"
                if states == {"complete"}
                else "invalid"
                if states == {"invalid"}
                else "incomplete"
            )
            response["results"] = payloads
            response["count"] = len(payloads)
            if static_graphs is not None:
                response["visualization"] = build_batch_runtime_attribution(
                    [payload["visualization"] for payload in payloads],
                    static_bundle=static_graphs,
                )
        return response

    def _scan_payload(
        self,
        scanner: DSLScanner,
        result,
        static_graphs: dict[str, Any] | None,
        *,
        node_ids: Mapping[str | None, str] | None = None,
    ) -> dict[str, Any]:
        """Serialize one fail-closed scan result for either tool mode.

        ``node_ids`` maps each decision-tree check to its node ID, so an explanation
        names the tree node each of its atoms belongs to."""

        with suppress(ConversationValidationError, IncompleteScanError):
            result.raise_for_status()
        payload: dict[str, Any] = {
            "completion_state": result.completion_state,
            "timings": {
                "total_seconds": result.scan_time,
                "rules": dict(result.rule_timings or {}),
            },
            "diagnostics": self._scan_diagnostics(result),
        }
        if static_graphs is not None:
            payload["visualization"] = build_runtime_attribution(
                static_graphs,
                result.rule_traces or {},
                matched_rules=set(result.rule_matches),
                completion_state=result.completion_state,
            )
        if result.is_complete:
            matched_rules = [scanner.loaded_rules[name] for name in result.rule_matches]
            payload["matched"] = bool(matched_rules)
            matches = []
            for rule in matched_rules:
                match = {"rule_name": rule.name, "metadata": dict(rule.metadata)}
                output = result.outputs.get(rule.name)
                if output is not None:
                    match["output"] = output.to_dict()
                matches.append(match)
            payload["matches"] = matches
            payload["explanations"] = [
                evidence.to_dict(node_ids) for evidence in (result.explanations or [])
            ]
        else:
            payload["partial_matches"] = list(result.rule_matches)
        return payload

    def _prepare_rule(
        self,
        rule_text: str,
        config_version: int | None,
    ) -> tuple[dict[str, Any], DSLScanner | None]:
        """Return the rule's compile report and scanner, compiling it once.

        The report is a copy, so a caller may change it. A scanner holds no per-scan
        state, so every scan of the rule reuses it.
        """

        if not isinstance(rule_text, str) or not rule_text.strip():
            raise PluginInputError("rule_text must be a non-empty string")
        if len(rule_text) > MAX_RULE_TEXT_CHARS:
            raise PluginInputError(f"rule_text must not exceed {MAX_RULE_TEXT_CHARS} characters")

        prepared = self._prepared.get_or_prepare(
            rule_text,
            runtime_identity=f"config-version:{config_version!r}",
            prepare=lambda source: self._compile_rule(source, config_version),
        )
        report, scanner = prepared
        return deepcopy(report), scanner

    def _compile_rule(
        self,
        rule_text: str,
        config_version: int | None,
    ) -> tuple[dict[str, Any], DSLScanner | None]:
        response = self._version_envelope()
        response["rule_hash"] = self._rule_hash(rule_text)
        response["warnings"] = self._deprecation_warnings_from_text(rule_text)

        try:
            config = self._rule_config(rule_text, config_version)
            # A batch reports every timeout per conversation, with its partial result, so the
            # re-raise that fail_on_timeout selects is turned off, as for QuarryGuard.
            scanner = DSLScanner(
                config,
                llm_rate_limiter=self._rate_limiter,
                limits=replace(config.get_safety_limits(), fail_on_timeout=False),
                _allow_parameters=True,
            )
            requirements = scanner.resource_requirements
            validation_warnings = []
            rule_complexities: dict[str, Complexity] = {}
            validator = RuleValidator()
            for name, rule in sorted(scanner.loaded_rules.items()):
                validation = validator.validate(rule)
                rule_complexities[name] = validation.estimated_complexity
                validation_warnings.extend(
                    {
                        "code": "validation_warning",
                        "severity": warning.severity.value,
                        "message": warning.message,
                        "suggestion": warning.suggestion,
                    }
                    for warning in validation.warnings
                )
                try:
                    advisory_warnings = [
                        {
                            "code": "validation_warning",
                            "severity": warning.severity.value,
                            "message": warning.message,
                            "suggestion": warning.suggestion,
                        }
                        for warning in quality_warnings(rule, scanner.compiled_rules[name])
                    ]
                except Exception:
                    # Advisory analysis must never turn a valid compilation into a failure.
                    advisory_warnings = []
                validation_warnings.extend(advisory_warnings)
            if not rule_complexities:
                raise RuleValidationError("Rule compilation produced no rules")
            estimated_complexity = max(rule_complexities.values(), key=_COMPLEXITY_RANK.__getitem__)
        except Exception as exc:
            diagnostic = self._exception_diagnostic(exc)
            requirements = getattr(exc, "requirements", {})
            response.update(
                {
                    "valid": False,
                    "stage": diagnostic["stage"],
                    "diagnostics": [diagnostic],
                    "required_capabilities": self._serialize_requirements(requirements),
                }
            )
            return response, None

        response["warnings"].extend(validation_warnings)
        response.update(
            {
                "valid": True,
                "stage": "complete",
                "rule_names": sorted(scanner.loaded_rules),
                # Hosts can read rule metadata before scanning; scan results include it
                # only when the rule matches.
                "rule_metadata": {
                    name: dict(rule.metadata) for name, rule in sorted(scanner.loaded_rules.items())
                },
                "diagnostics": [],
                "required_capabilities": self._serialize_requirements(requirements),
                "estimated_complexity": estimated_complexity.value,
                "rule_complexities": {
                    name: complexity.value for name, complexity in rule_complexities.items()
                },
                "parameter_schema": [
                    {"name": spec.name, "type": spec.type, "binding": spec.binding}
                    for artifact in scanner.compiled_rules.values()
                    for spec in artifact.parameter_schema
                ],
                # A rule's checks and their edges, from its compiled artifact; a text with
                # several rules is scanned as a whole and has none.
                "decision_tree": (
                    {
                        name: tree_report(artifact, scanner_digests(scanner, self._evidence_key))
                        for name, artifact in scanner.compiled_rules.items()
                    }
                    if len(scanner.compiled_rules) == 1
                    else {}
                ),
                "visualization": build_visualization_bundle(
                    scanner.loaded_rules, scanner.compiled_rules
                ).to_dict(),
            }
        )
        return response, scanner

    def _rule_config(self, rule_text: str, config_version: int | None) -> QuarryConfig:
        if config_version is not None and (
            isinstance(config_version, bool) or config_version not in {1, 2}
        ):
            raise PluginInputError("config_version must be 1 or 2")
        if self._runtime_config is None:
            return QuarryConfig(
                quarry_rule_texts=(rule_text,),
                config_version=config_version or 1,
            )
        if config_version is not None and config_version != self._runtime_config.config_version:
            raise PluginInputError(
                "config_version does not match the trusted server configuration "
                f"({self._runtime_config.config_version})"
            )
        # The trusted configuration is deeply immutable, so the per-request variant shares its
        # models, prompts, and limits by reference — only the rule source differs.
        return replace(
            self._runtime_config,
            quarry_rules_paths=(),
            quarry_rule_texts=(rule_text,),
        )

    def _index_entries(self, kinds: set[str]) -> list[dict[str, Any]]:
        entries = []
        for item in self._catalog["items"]:
            if kinds and item["kind"] not in kinds:
                continue
            entries.append(self._compact_item(item))
        if not kinds or "example" in kinds:
            entries.extend(self._compact_example(example) for example in self._examples.values())
        return entries

    def _rejected_entries(self, query: str | None) -> list[dict[str, Any]]:
        """Project the rejected-spelling table, optionally narrowed by a substring."""

        needle = query.strip().lower() if query else None
        entries = []
        for record in self._catalog["rejected_spellings"]:
            haystack = " ".join(
                value for value in record.values() if isinstance(value, str)
            ).lower()
            if needle is not None and needle not in haystack:
                continue
            # Absent keys read as "not applicable" and cost the caller nothing. That is the
            # same rule every other record follows, so it is applied from the same place.
            entries.append(_prune(record))
        return entries

    def _search(self, query: str, kinds: set[str]) -> list[dict[str, Any]]:
        query_tokens = self._tokens(query)
        if not query_tokens:
            raise PluginInputError("query must contain at least one searchable word or number")
        normalized_query = " ".join(sorted(query_tokens))

        # Index calls once per search. Example rules are packaged and tested, but matching
        # their raw text would mistake quoted DSL strings and comments for invocations.
        callable_names = {
            name
            for item in self._catalog["items"]
            if item["kind"] in _CALLABLE_KINDS
            if (name := item.get("name") or item.get("spelling"))
        }
        call_examples: dict[str, list[str]] = defaultdict(list)
        for example in self._examples.values():
            for name in _rule_call_names(example["rule_text"], callable_names):
                call_examples[name].append(example["rule_text"])

        scored = []
        for item in self._catalog["items"]:
            if kinds and item["kind"] not in kinds:
                continue
            candidate = dict(item)
            # An example may use a function without listing it among its teaching targets.
            # Only callable records receive the invocation signal. Flow words and block
            # keywords can precede parentheses without being function calls.
            name = item.get("name") or item.get("spelling")
            if item["kind"] in _CALLABLE_KINDS and name:
                candidate["rule_text"] = "\n".join(call_examples.get(name, ()))
            candidate["example_text"] = "\n".join(
                example["rule_text"]
                for example in self._examples.values()
                if item["id"] in example["covered_item_ids"]
            )
            score, matched_fields = self._score(candidate, query_tokens, normalized_query)
            if score:
                result = self._compact_item(item)
                result.update({"score": score, "matched_fields": matched_fields})
                scored.append(result)

        if not kinds or "example" in kinds:
            for example in self._examples.values():
                candidate = {
                    "id": example["id"],
                    "name": example["id"].removeprefix("example:"),
                    "summary": "Tested Quarry example",
                    "rule_text": example["rule_text"],
                    "example_text": example["search_text"],
                }
                score, matched_fields = self._score(candidate, query_tokens, normalized_query)
                if score:
                    result = self._compact_example(example)
                    result.update({"score": score, "matched_fields": matched_fields})
                    scored.append(result)

        kind_order = {"function": 0, "compiler_form": 0, "example": 1, "token": 3}
        ranked = sorted(
            scored,
            key=lambda value: (
                -value["score"],
                kind_order.get(value["kind"], 2),
                value["id"],
            ),
        )
        # Leave room for a nearly-as-relevant worked example in the default top ten.
        # Do not promote a weak example just to satisfy a fixed kind quota.
        if len(ranked) > 10 and not any(item["kind"] == "example" for item in ranked[:10]):
            cutoff = ranked[9]["score"]
            example = next(
                (
                    item
                    for item in ranked[10:]
                    if item["kind"] == "example" and item["score"] * 4 >= cutoff * 3
                ),
                None,
            )
            if example is not None:
                ranked.remove(example)
                ranked.insert(9, example)
        return ranked

    def _score(
        self,
        candidate: dict[str, Any],
        query_tokens: set[str],
        normalized_query: str,
    ) -> tuple[int, list[str]]:
        # A word repeated in an id, name, signature, and description is one piece of
        # evidence. Its strongest field determines its score; additional fields remain
        # visible in matched_fields without multiplying that evidence.
        token_weights: dict[str, int] = {}
        matched_fields = []
        for field, weight in _SEARCH_FIELDS:
            # A helper mentioned in a large example is not relevant solely because an
            # unrelated phrase appears elsewhere in that rule. Require at least one
            # match on the helper's own fields before using example text as recall.
            if (
                field in {"rule_text", "example_text"}
                and candidate.get("kind") in _CALLABLE_KINDS
                and not token_weights
            ):
                continue
            value = candidate.get(field)
            if value is None:
                continue
            text = json.dumps(value, sort_keys=True) if not isinstance(value, str) else value
            field_tokens = self._tokens(text)
            overlap = query_tokens & self._indexed_tokens(field_tokens)
            if not overlap:
                continue
            matched_fields.append(field)
            for token in overlap:
                token_weights[token] = max(token_weights.get(token, 0), weight)
            # Identity compares both sides as written: index expansion widens what a query
            # can reach, never what counts as having asked for exactly this field.
            normalized_value = " ".join(sorted(field_tokens))
            if normalized_query and normalized_query == normalized_value:
                for token in overlap:
                    token_weights[token] = max(token_weights[token], weight * 10)
        # A complete multiword identifier match is a strong route to its catalog record.
        # This applies equally to functions, constructs, surfaces, and examples.
        identity = candidate.get("name") or candidate.get("id", "")
        identity_tokens = self._tokens(identity)
        identity_bonus = 100 if len(identity_tokens) >= 2 and identity_tokens <= query_tokens else 0
        score = sum(token_weights.values()) + identity_bonus
        return score, matched_fields

    def _full_record(
        self, record_id: str, verbosity: ReferenceVerbosity = "compact"
    ) -> dict[str, Any] | None:
        # ``full`` is the ground-truth view: it keeps every empty and null field, and every
        # example's own coverage list, so an author who asked to see everything can tell a
        # field that does not exist from one that was simply not set. ``compact`` prunes those
        # absences for its budget. Compiler-only bookkeeping is dropped from an item in both
        # views -- an author cannot write it and an agent cannot act on it -- so ``full`` means
        # "every field the language reports", not "every field the catalog stores".
        full = verbosity == "full"
        finish = deepcopy if full else _prune
        if record_id in self._examples:
            return finish(self._example_record(self._examples[record_id], verbosity))
        item = self._items.get(record_id)
        if item is None:
            return None
        record = {key: value for key, value in item.items() if key not in _INTERNAL_RECORD_KEYS}
        record["availability"] = self._availability(item)
        record["examples"] = [
            self._example_record(example, verbosity, include_coverage=full)
            for example in self._examples.values()
            if record_id in example["covered_item_ids"]
        ]
        return finish(record)

    @staticmethod
    def _example_record(
        example: dict[str, Any],
        verbosity: ReferenceVerbosity,
        *,
        include_coverage: bool = True,
    ) -> dict[str, Any]:
        """Project one tested example, spelling out its conversations only when asked.

        A check's conversation is the bulkiest thing the catalog holds and the one thing an
        author rarely reads: the rule text and whether the check is meant to match already say
        what the example proves. ``full`` includes it, and ``input_path`` still names it either
        way, so asking for the example itself always reaches the conversation.

        ``include_coverage`` is false for an example reached through a record it covers. The
        list of everything else it covers answers a question the caller did not ask, and a
        popular example repeats it under every record in the response; ``get`` on the example's
        own id is where that list belongs.
        """

        record = dict(example)
        record.pop("search_text", None)
        record["kind"] = "example"
        if not include_coverage:
            record.pop("covered_item_ids", None)
        if verbosity == "full":
            return record
        record["checks"] = [
            {key: value for key, value in check.items() if key != "input"}
            for check in record["checks"]
        ]
        return record

    def _compact_item(self, item: dict[str, Any]) -> dict[str, Any]:
        return _prune(
            {
                "id": item["id"],
                "kind": item["kind"],
                "name": item.get("name", item.get("spelling")),
                "signature": item.get("signature"),
                "summary": item["summary"],
                "deprecated": item.get("deprecated", False),
                "availability": self._availability(item),
            }
        )

    @staticmethod
    def _compact_example(example: dict[str, Any]) -> dict[str, Any]:
        return _prune(
            {
                "id": example["id"],
                "kind": "example",
                "name": example["id"].removeprefix("example:"),
                "summary": "Tested Quarry example",
                "requires_backend": example["requires_backend"],
                "covered_item_ids": list(example["covered_item_ids"]),
            }
        )

    def _materialize_example(self, record: dict[str, Any]) -> dict[str, Any]:
        result = dict(record)
        rule_text = self._read_language_asset(record["rule_path"])
        checks = []
        search_parts = [rule_text]
        for check in record["checks"]:
            raw_input = self._read_language_asset(check["input_path"])
            conversation = json.loads(raw_input)
            checks.append(
                {
                    "input_path": check["input_path"],
                    "input": conversation,
                    "expected_match": check["expected_match"],
                }
            )
            search_parts.append(raw_input)
        result.update(
            {
                "rule_text": rule_text,
                "checks": checks,
                "search_text": "\n".join(search_parts),
            }
        )
        return result

    @staticmethod
    def _read_language_asset(relative_path: str) -> str:
        path = relative_path.removeprefix("examples/")
        return (
            resources.files("quarry.language")
            .joinpath("examples", path)
            .read_text(encoding="utf-8")
        )

    def _availability(self, item: dict[str, Any]) -> dict[str, Any]:
        group = item.get("resource_group")
        if group is None:
            return {"status": "available", "resource_group": None}
        # A new catalog group is unavailable until this adapter learns how to verify its
        # server configuration. Returning that conservative state keeps reference lookup
        # usable while preventing an unknown capability from being advertised as ready.
        status = self._runtime_capabilities().get(group, "unconfigured")
        return {
            "status": status,
            "resource_group": group,
            "reason": (
                "Trusted server configuration is present; compile a complete rule to verify "
                "the selected alias and adapter operation."
                if status == "requires_compile"
                else "This capability requires trusted server configuration."
            ),
        }

    def _runtime_capabilities(self) -> dict[str, str]:
        config = self._runtime_config
        models = list(config.models.values()) if config is not None else []
        configured = {
            "model": bool(models),
            "score": any(model.score is not None for model in models),
            "completion": bool(models),
            "token": any(model.tokens is not None for model in models),
            "topic_prober": bool(config and config.allow_stub_probers),
            "capability_classifier": bool(config and config.allow_stub_probers),
        }
        return {
            group: "requires_compile" if is_configured else "unconfigured"
            for group, is_configured in configured.items()
        }

    def _deprecation_warnings_from_text(self, rule_text: str) -> list[dict[str, Any]]:
        warnings = []
        executable_text = self._executable_text(rule_text)
        for item in self._catalog["items"]:
            if not item.get("deprecated") or not item.get("name"):
                continue
            name = item["name"]
            pattern = rf"(?<![A-Za-z0-9_.]){re.escape(name)}\s*\("
            if re.search(pattern, executable_text):
                warnings.append(
                    {
                        "code": "deprecated_language_item",
                        "severity": "warning",
                        "item_id": item["id"],
                        "message": f"{name} is deprecated.",
                        "replacement": item.get("replacement"),
                    }
                )
        return warnings

    @staticmethod
    def _executable_text(rule_text: str) -> str:
        """Mask quoted strings and line comments before checking language-item use."""

        characters = list(rule_text)
        in_string = False
        in_comment = False
        escaped = False
        index = 0
        while index < len(characters):
            character = characters[index]
            if in_comment:
                if character in "\r\n":
                    in_comment = False
                else:
                    characters[index] = " "
                index += 1
                continue
            if in_string:
                characters[index] = " "
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
                index += 1
                continue
            if character == '"':
                in_string = True
                characters[index] = " "
            elif character == "/" and index + 1 < len(characters) and characters[index + 1] == "/":
                in_comment = True
                characters[index] = " "
                characters[index + 1] = " "
                index += 1
            index += 1
        return "".join(characters)

    def _serialize_requirements(self, requirements: dict[str, set[Any]]) -> dict[str, Any]:
        # Resource analysis also sees compiler-generated helpers such as ``_index``.
        # Only catalogued language names are part of the public plugin response.
        functions = {
            name for name in requirements.get("functions", set()) if name in self._items_by_name
        }
        model_operations = sorted(requirements.get("model_operations", set()))
        resource_groups = {
            self._items_by_name[name].get("resource_group")
            for name in functions
            if name in self._items_by_name
        }
        resource_groups.update(
            self._items_by_name[f"model.{operation}"].get("resource_group")
            for _, operation in model_operations
            if f"model.{operation}" in self._items_by_name
        )
        return {
            "resource_groups": sorted(group for group in resource_groups if group),
            "functions": sorted(functions),
            "model_operations": [
                {"model": model, "operation": operation} for model, operation in model_operations
            ],
            "models": sorted(requirements.get("llm_models", set())),
            "text_resources": sorted(requirements.get("text_resources", set())),
        }

    @staticmethod
    def _scan_diagnostics(result) -> list[dict[str, Any]]:
        diagnostics = []
        if result.validation_error:
            diagnostics.append({"code": "invalid_conversation", "message": result.validation_error})
        if result.error and not result.validation_error:
            diagnostics.append({"code": "scan_error", "message": result.error})
        for name in result.timed_out_rules or []:
            diagnostics.append({"code": "rule_timeout", "rule_name": name})
        for name in result.skipped_rules or []:
            diagnostics.append({"code": "rule_skipped", "rule_name": name})
        for name, message in (result.evaluation_errors or {}).items():
            diagnostics.append({"code": "evaluation_error", "rule_name": name, "message": message})
        if result.was_truncated:
            diagnostics.append(
                {"code": "conversation_truncated", "details": result.truncation_info or {}}
            )
        diagnostics.extend(
            {"code": "evaluation_warning", "message": message}
            for message in (result.evaluation_warnings or [])
        )
        return diagnostics

    @staticmethod
    def _exception_diagnostic(exc: Exception) -> dict[str, Any]:
        stage = "internal"
        code = type(exc).__name__
        details = getattr(exc, "details", {})
        if isinstance(exc, PluginInputError):
            stage = "input"
        elif isinstance(exc, RuleParseError):
            stage = "parse"
        elif isinstance(exc, RuleValidationError):
            error_type = details.get("error_type") if isinstance(details, dict) else None
            if error_type == "unknown_function":
                stage = "link"
            elif "could not be compiled" in str(exc):
                stage = "compile"
            else:
                stage = "validate"
        elif isinstance(exc, (ConfigurationError,)):
            stage = "configure"
        elif isinstance(exc, (ResourceError, ResourcePreparationError, ModuleNotFoundError)):
            stage = "resources"
        elif isinstance(exc, QuarryError):
            stage = "evaluate"
        diagnostic = {
            "code": code,
            "severity": "error",
            "stage": stage,
            "message": str(exc),
        }
        if details:
            diagnostic["details"] = details
        line = getattr(exc, "line", None)
        if line is not None:
            diagnostic["line"] = line
        return diagnostic

    def _version_envelope(self) -> dict[str, Any]:
        return {
            "plugin_version": PLUGIN_VERSION,
            "quarry_version": QUARRY_VERSION,
            "catalog_format_version": self._catalog["format_version"],
            "language_version": self._catalog["language_version"],
            "compiler_version": self._catalog["compiler_version"],
        }

    @staticmethod
    def _rule_hash(rule_text: str) -> str:
        return hashlib.sha256(rule_text.encode("utf-8")).hexdigest()

    @staticmethod
    def _tokens(value: str) -> set[str]:
        """Tokenize text exactly as it is written, widening nothing."""

        tokens = set()
        for token in _WORD.findall(value.lower()):
            token = _TOKEN_ALIASES.get(token, token)
            if token in _STOP_WORDS:
                continue
            tokens.add(token)
        return tokens

    @staticmethod
    def _indexed_tokens(tokens: set[str]) -> set[str]:
        """Add the searchable segment of every separator-free compound in ``tokens``.

        Expansion belongs to the index and never to the query. A query is what the author
        actually asked for, so widening it too would make an exact search for ``fullword``
        reach every record that merely mentions a "word" and rank unrelated syntax words
        alongside the one thing that was named."""

        expanded = set(tokens)
        for token in tokens:
            expanded.update(_COMPOUND_TOKENS.get(token, ()))
        return expanded
