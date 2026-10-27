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

"""MCP transport for the host-neutral Quarry plugin service."""

from __future__ import annotations

import asyncio
import io
import os
import sys
from typing import Annotated, Any, Literal, cast

import pydantic_core
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError
from mcp.types import CallToolResult, InputRequiredResult, TextContent, ToolAnnotations
from pydantic import Field, StrictFloat, StrictInt, StrictStr

from quarry.decision_report import decision_flows
from quarry.plugin.service import (
    MAX_FLOW_LABEL_CHARS,
    MAX_FLOW_PATHS,
    MAX_PARAMETER_TEXT_CHARS,
    MAX_REFERENCE_ID_CHARS,
    MAX_REFERENCE_IDS,
    MAX_REFERENCE_QUERY_CHARS,
    MAX_RULE_TEXT_CHARS,
    MAX_SCAN_BATCH,
    QuarryPlugin,
    ReferenceKind,
    load_runtime_config,
)

ParameterValue = (
    StrictInt
    | Annotated[StrictFloat, Field(allow_inf_nan=False)]
    | Annotated[StrictStr, Field(max_length=MAX_PARAMETER_TEXT_CHARS)]
)


class StrictMCPServer(MCPServer):
    """Reject unknown tool fields and publish that restriction in each JSON schema."""

    async def list_tools(self):
        tools = await super().list_tools()
        for tool in tools:
            tool.input_schema["additionalProperties"] = False
        return tools

    async def call_tool(
        self, name: str, arguments: dict[str, Any], context: Context | None = None
    ) -> CallToolResult | InputRequiredResult:
        tools = {tool.name: tool for tool in await self.list_tools()}
        if name not in tools:
            raise ToolError(f"Unknown tool: {name}")
        allowed = set(tools[name].input_schema.get("properties", {}))
        unknown = sorted(set(arguments) - allowed)
        if unknown:
            raise ToolError(f"Unknown arguments for {name}: {', '.join(unknown)}")
        try:
            result = await super().call_tool(name, arguments, context)
        except UnexpectedToolError as exc:
            # MCPServer hides every tool exception's text. A ValueError is a rejected
            # request, and its message tells the agent what to fix; other crashes stay hidden.
            if isinstance(exc.__cause__, ValueError):
                raise ToolError(f"{exc}: {exc.__cause__}") from exc.__cause__
            raise
        return _compact_text_content(result)


def _compact_text_content(
    result: CallToolResult | InputRequiredResult,
) -> CallToolResult | InputRequiredResult:
    """Re-emit a tool's text content as compact JSON, matching the reference budget.

    MCPServer pretty-prints a tool's ``dict`` return into the text block a client reads,
    indenting it two spaces per level. That whitespace is roughly a fifth of the tokens an
    agent spends on a reference response and carries nothing it can act on, so the response
    the acceptance budget measures is far larger than the data in it. When structured content
    accompanies the text -- which it does for every tool here, all of which return a mapping
    -- rebuild the text from that same structured value with no indentation. The structured
    content is left untouched, and any result without structured content (an error, say)
    passes through unchanged.
    """

    if isinstance(result, CallToolResult) and result.structured_content is not None:
        text = pydantic_core.to_json(result.structured_content, fallback=str).decode()
        return result.model_copy(update={"content": [TextContent(type="text", text=text)]})
    return result


_config_path = os.environ.get("QUARRY_PLUGIN_CONFIG") or None
_service = QuarryPlugin(load_runtime_config(_config_path))
mcp = StrictMCPServer("Quarry")
_LOCAL_READ_ONLY = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)
_SCAN_READ_ONLY = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=False,
    open_world_hint=True,
)


@mcp.tool(annotations=_LOCAL_READ_ONLY)
async def quarry_reference(
    action: Literal["index", "search", "get", "rejected"],
    query: Annotated[str, Field(max_length=MAX_REFERENCE_QUERY_CHARS)] | None = None,
    ids: Annotated[
        list[Annotated[str, Field(min_length=1, max_length=MAX_REFERENCE_ID_CHARS)]],
        Field(max_length=MAX_REFERENCE_IDS),
    ]
    | None = None,
    kinds: Annotated[list[ReferenceKind], Field(max_length=6)] | None = None,
    limit: Annotated[int, Field(ge=1, le=50, strict=True)] | None = None,
    offset: Annotated[int, Field(ge=0, strict=True)] = 0,
    verbosity: Literal["compact", "full"] | None = None,
    include_precedence: bool | None = None,
) -> dict[str, Any]:
    """Discover the installed Quarry language.

    Use ``search`` with plain words to find relevant features, ``get`` with exact IDs
    to read full descriptions and tested examples, or ``index`` to browse compact records.
    Continue an index with the returned ``next_offset`` value.

    ``limit`` paginates ``index`` and ``search`` and defaults to 10 for them; ``get`` and
    ``rejected`` take no ``limit`` and reject one that is supplied. A ``get`` already names
    exactly what it wants, so ask for as many IDs as its schema bound allows in one call,
    splitting a longer list across calls, and account for all of them from the response:
    ``missing_ids`` means no such record, while ``filtered_ids`` means the ID was right and
    the request's own ``kinds`` excluded it, so widen that filter instead of searching again.

    Use ``rejected`` to read the spellings Quarry refuses and what to write instead. It
    returns the complete table, or only matching rows when a ``query`` substring is given.

    Every record omits keys that do not apply to it, so a missing key means "not applicable",
    never "withheld". ``get`` accepts ``verbosity``: the default ``compact`` prunes those empty
    and null fields, names each tested example, and says whether it is meant to match, while
    ``full`` is the ground-truth view -- it keeps every empty and null field, each example's own
    coverage list, and spells out the conversation each check scans, so a field that does not
    exist can be told from one that was simply not set. The operator-precedence table accompanies
    ``index``; pass ``include_precedence=true`` to read it alongside any other action.
    """

    return await asyncio.to_thread(
        _service.reference,
        action=action,
        query=query,
        ids=ids,
        kinds=kinds,
        limit=limit,
        offset=offset,
        verbosity=verbosity,
        include_precedence=include_precedence,
    )


@mcp.tool(annotations=_LOCAL_READ_ONLY)
async def quarry_compile(
    rule_text: Annotated[str, Field(max_length=MAX_RULE_TEXT_CHARS)],
    config_version: Literal[1, 2] | None = None,
    visualize: bool = False,
) -> dict[str, Any]:
    """Validate and compile one complete Quarry rule without scanning data.

    The response identifies the failed stage, structured diagnostics, deprecations,
    rule metadata, parameters, and required runtime capabilities. ``visualize`` adds the
    rule's static source graph. The decision tree is ``quarry_decision_tree``'s. Runtime
    configuration is controlled by the server.
    """

    return await asyncio.to_thread(
        _service.compile,
        rule_text=rule_text,
        config_version=config_version,
        visualize=visualize,
    )


@mcp.tool(annotations=_LOCAL_READ_ONLY)
async def quarry_decision_tree(
    rule_text: Annotated[str, Field(max_length=MAX_RULE_TEXT_CHARS)],
    config_version: Literal[1, 2] | None = None,
) -> dict[str, Any]:
    """Return one Quarry rule's decision tree without scanning data.

    Each node is one check of the rule, keyed by a stable ID, with its label, its kind
    (check, score, number or collect), and the step or leaf (match, no_match) each outcome
    leads to. A score node adds its comparison and a threshold-free score key; a number
    node (a collection size such as ``len(messages) > 5``) adds its comparison. A scan's
    ``decision_path`` names the nodes it reached and the tree's ``tree_sha256``.
    """

    return await asyncio.to_thread(
        _service.decision_tree,
        rule_text=rule_text,
        config_version=config_version,
    )


@mcp.tool(annotations=_LOCAL_READ_ONLY)
async def quarry_decision_flows(
    tree: dict[str, Any],
    paths: Annotated[list[dict[str, Any]], Field(min_length=1, max_length=MAX_FLOW_PATHS)],
    labels: list[Annotated[str, Field(max_length=MAX_FLOW_LABEL_CHARS)]] | None = None,
    thresholds: dict[str, Annotated[float, Field(allow_inf_nan=False)]] | None = None,
) -> dict[str, Any]:
    """Count scans through one rule's decision tree, optionally at moved thresholds.

    ``tree`` is the rule's decision tree and ``paths`` the ``decision_path`` of each scan
    of it; every path must carry the tree's ``tree_sha256``. ``labels`` gives one label per
    path, such as its ground truth. ``thresholds`` moves score or number thresholds, keyed
    by node ID or by parameter name, and re-walks each path over its recorded values
    without scanning again. Returns the scans by outcome and label at each step, and by
    label at each leaf: match, no_match, unknown (a moved threshold reached a check the
    scan never answered) and error (the scan failed at a check).
    """

    return await asyncio.to_thread(
        decision_flows, tree, paths, labels=labels, thresholds=thresholds
    )


@mcp.tool(annotations=_SCAN_READ_ONLY)
async def quarry_scan(
    rule_text: Annotated[str, Field(max_length=MAX_RULE_TEXT_CHARS)],
    conversation: dict[str, Any] | None = None,
    conversations: Annotated[list[dict[str, Any]], Field(min_length=1, max_length=MAX_SCAN_BATCH)]
    | None = None,
    parameters: dict[str, ParameterValue] | None = None,
    explain: bool | Literal["support"] = True,
    visualize: bool = False,
    config_version: Literal[1, 2] | None = None,
    explore_rate: Annotated[float, Field(ge=0, le=1)] = 0.0,
) -> dict[str, Any]:
    """Compile one Quarry rule once and scan one conversation or an ordered batch.

    Supply every value declared by the rule in one ``parameters`` map. Unmarked declarations
    are scan-bound; declarations marked ``policy`` are fixed on the ad hoc compiled policy.
    A clean match decision is returned only for a complete scan. Invalid, truncated, timed-out,
    skipped, or failed evaluation remains explicit and fails closed. Set ``visualize`` only
    when drawing the rule graph; it returns per-check runtime data that grows with the
    conversation. ``explain`` true explains every atom of the rule, matched or not, which
    can call models; ``"support"`` returns only a match's minimal support, with no
    further evaluation.

    Each complete result of a one-rule text has a ``decision_path``: the conversation's
    SHA-256, the tree nodes it reached with their outcomes and model scores, and its leaf.
    ``explore_rate`` is the fraction of conversations, selected by that hash, whose path
    also scores the nodes it skipped.

    Scans reuse the model results this server already has for the same model and input,
    so a rescan, or one that moves only a threshold, makes no new model call. A result
    expires ``model_cache.ttl_seconds`` (default 12 hours) after its last use.
    """

    return await _service.scan(
        rule_text=rule_text,
        conversation=conversation,
        conversations=conversations,
        parameters=parameters,
        explain=explain,
        visualize=visualize,
        config_version=config_version,
        explore_rate=explore_rate,
    )


def main() -> None:
    """Run the Quarry MCP server over standard input and output."""

    # While it serves, the stdio transport points descriptor 1 at stderr and writes JSON-RPC
    # to a private copy, so a library's print() misses the wire. Python buffers sys.stdout,
    # though, and text left in that buffer reaches the restored descriptor at exit, after the
    # last JSON-RPC message. LiteLLM prints a help banner for each failed call, for example.
    # Line buffering flushes each print while descriptor 1 still points at stderr.
    cast(io.TextIOWrapper, sys.stdout).reconfigure(line_buffering=True)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
