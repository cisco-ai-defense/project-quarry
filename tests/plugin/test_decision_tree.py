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

"""Decision trees and decision paths through the plugin's public compile and scan."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from quarry import compile_policy, load_compiled_policy
from quarry.config import QuarryConfig
from quarry.decision_report import telemetry_record
from quarry.plugin.server import mcp
from quarry.plugin.service import QuarryPlugin
from quarry.types.exceptions import LLMError

SYSTEM = {"role": "system", "content": "You are a writing assistant with a web search tool."}

FICTION = {
    "messages": [
        SYSTEM,
        {"role": "user", "content": "Help me write a story about a detective in Paris."},
        {"role": "assistant", "content": "The detective walks along the Seine at dawn."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"q": "Paris poisons 1900"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "Arsenic was a common poison."},
        {"role": "user", "content": "Now the villain threatens to kill the witness."},
        {"role": "assistant", "content": "The villain steps out of the shadows."},
    ]
}
BENIGN_STORY = {
    "messages": [
        SYSTEM,
        {"role": "user", "content": "Write a short story about a cat who loves gardens."},
        {"role": "assistant", "content": "Milo spent his mornings among the tomatoes."},
        {"role": "user", "content": "Add a chapter where he meets a hedgehog."},
        {"role": "assistant", "content": "One evening a hedgehog shuffled under the fence."},
    ]
}
NO_STORY = {
    "messages": [
        SYSTEM,
        {"role": "user", "content": "How do I kill a stuck process with poison pills?"},
    ]
}


class _WordTokens:
    """Count one token per word plus four per message, like a chat template would."""

    async def tokenize(self, config, messages, tools, *, timeout=None):
        words = sum(len(str(message.get("content") or "").split()) for message in messages)
        return {config.tokens.count_field: words + 4 * len(messages)}


_WORDS = ("kill", "villain", "poison")


class _KeywordJudge:
    """Score the scored text by its violent words, deterministically."""

    async def score(self, config, messages, *, timeout=None):
        return _keyword_score(" ".join(str(message.get("content") or "") for message in messages))


def _rule(threshold: float, name: str = "Exempt violence in fiction") -> str:
    return f'''rule "{name}"
metadata:
    category = "fiction"
condition:
    (any chunk in model("judge").split(messages, 40) where
        model("judge").score(chunk, "fiction") > {threshold})
    and (any m in user_messages where m.content contains "story")
'''


@pytest.fixture
def plugin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[QuarryPlugin, _KeywordJudge]:
    async def run_inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    judge = _KeywordJudge()
    monkeypatch.setattr(asyncio, "to_thread", run_inline)
    monkeypatch.setattr(
        "quarry.integrations.model_inference._default_adapters",
        lambda: {"judge": judge, "words": _WordTokens()},
    )
    config = QuarryConfig.from_mapping(
        {
            "config_version": 2,
            # A decision path scans each node too; the default provider pacing would
            # make this test wait out whole rate-limit intervals.
            "llm_rate_limits": {"default": {"requests_per_interval": None, "max_concurrent": 8}},
            "models": {
                "judge": {
                    "adapter": "judge",
                    "model": "keyword-judge",
                    "tokens": {"adapter": "words", "extraction": {"count": "count"}},
                    "score": {
                        "message_builder": "constitution",
                        "boundary_seed": 7,
                        "tokens": {"negative": "0", "positive": "1"},
                    },
                }
            },
        },
        tmp_path,
        require_rules=False,
    )
    return QuarryPlugin(config), judge


# A fetched page far longer than one 40-token chunk, with the violent passage at its end.
PAGE = " ".join(["The museum opens at nine and closes at six."] * 12) + (
    " The villain hid the poison in the gallery."
)
LONG_TOOL = {
    "messages": [
        SYSTEM,
        {"role": "user", "content": "Summarize this page for my story."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "fetch", "arguments": '{"url": "https://m.example"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": PAGE},
    ]
}


def _keyword_score(text: str) -> float:
    return min(1.0, 0.1 + 0.3 * sum(text.lower().count(word) for word in _WORDS))


def _span_text(conversation: dict, span: dict) -> str:
    """Return the text a span covers: the tail of its first message to the head of its last."""

    first = span["message_index"]
    last = span.get("end_message_index", first)
    texts = []
    for index in range(first, last + 1):
        content = conversation["messages"][index]["content"]
        texts.append(
            content[span["start"] if index == first else 0 : span["end"] if index == last else None]
        )
    return " ".join(texts)


def _chunk_text(conversation: dict, chunk: dict) -> str:
    spans = [chunk["span"]] if chunk["context"] is None else [chunk["context"], chunk["span"]]
    return " ".join(_span_text(conversation, span) for span in spans)


def _whole(conversation: dict, first: int, last: int) -> dict:
    """Return the span of messages ``first`` to ``last`` of a conversation, whole."""

    return {
        "message_index": first,
        "start": 0,
        "end_message_index": last,
        "end": len(conversation["messages"][last]["content"]),
    }


@pytest.mark.asyncio
async def test_scan_returns_each_conversations_path_through_the_compiled_tree(plugin):
    plugin, judge = plugin
    rule = _rule(0.6)

    # The compile report holds the tree: the local check first, then the split score.
    (tree,) = plugin.decision_tree(rule_text=rule)["decision_tree"].values()
    story_id, score_id = tree["root"], tree["nodes"][tree["root"]]["true"]
    story, score = tree["nodes"][story_id], tree["nodes"][score_id]
    assert (story["kind"], story["true"], story["false"]) == ("check", score_id, "no_match")
    assert story["label"] == 'any m in user_messages where (m.content contains "story")'
    assert score["kind"] == "score"
    assert score["score_form"] == {"quantifier": "any", "operator": ">", "threshold": 0.6}
    assert (score["true"], score["false"]) == ("match", "no_match")

    batch = [FICTION, BENIGN_STORY, NO_STORY, LONG_TOOL]
    hashes = [
        hashlib.sha256(
            json.dumps(conversation, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        for conversation in batch
    ]
    samples = [int(digest[:16], 16) / 2**64 for digest in hashes]
    rate = sorted(samples)[2]  # explores the two lowest-sampled conversations
    result = await plugin.scan(rule_text=rule, conversations=batch, explore_rate=rate)
    fiction, benign, no_story, long_tool = result["results"]

    assert [item["matched"] for item in result["results"]] == [True, False, False, True]
    assert all(item["diagnostics"] == [] for item in result["results"])
    for item, digest, sample in zip(result["results"], hashes, samples, strict=True):
        path = item["decision_path"]
        # The path walks the compiled tree from its root to the verdict's leaf, and names
        # the tree, the configuration and the parameter values it was scanned with.
        key = tree["root"]
        for entry in path["path"]:
            assert entry["node"] == key
            key = tree["nodes"][key]["true" if entry["outcome"] else "false"]
        assert key == path["leaf"] == ("match" if item["matched"] else "no_match")
        assert path["tree_sha256"] == tree["tree_sha256"]
        assert path["config_sha256"] == result["results"][0]["decision_path"]["config_sha256"]
        assert path["parameter_digest"] == result["parameter_digest"]
        assert path["conversation_sha256"] == digest
        assert path["exploration"]["sample_value"] == sample
        assert path["exploration"]["selected"] is (sample < rate)

    # The chunks partition the conversation in order and repeat nothing: each chunk is one
    # span, and its score is the judge's score of exactly the text that span covers.
    step = fiction["decision_path"]["path"][1]
    assert step["node"] == score_id and step["outcome"] is True
    decision = step["score"]
    chunks = decision["chunks"]
    assert [chunk["span"] for chunk in chunks] == [_whole(FICTION, 0, 2), _whole(FICTION, 3, 6)]
    assert all(chunk["context"] is None for chunk in chunks)
    assert [chunk["value"] for chunk in chunks] == [
        pytest.approx(_keyword_score(_chunk_text(FICTION, chunk))) for chunk in chunks
    ]
    assert decision["evaluations"] == len(chunks)
    assert decision["value"] == max(chunk["value"] for chunk in chunks)
    best = max(range(len(chunks)), key=lambda ordinal: (chunks[ordinal]["value"], -ordinal))
    assert decision["deciding"] == {key: chunks[best][key] for key in ("chunk", "span", "context")}
    assert decision["matching"] == sum(chunk["value"] > 0.6 for chunk in chunks)

    # A tool response longer than one chunk is cut into ranges of equal length at
    # whitespace, the spec's example: the first chunk ends at the tool call, each range is
    # its own chunk, and the range holding "villain" and "poison" is the detected span.
    long_score = long_tool["decision_path"]["path"][1]["score"]
    assert [chunk["span"] for chunk in long_score["chunks"]] == [
        {"message_index": 0, "start": 0, "end_message_index": 2, "end": 0},
        {"message_index": 3, "start": 0, "end": 142},
        {"message_index": 3, "start": 142, "end": 283},
        {"message_index": 3, "start": 283, "end": 424},
        {"message_index": 3, "start": 424, "end": 570},
    ]
    assert [chunk["value"] for chunk in long_score["chunks"]] == [
        pytest.approx(_keyword_score(_chunk_text(LONG_TOOL, chunk)))
        for chunk in long_score["chunks"]
    ]
    assert long_score["deciding"] == {
        "chunk": 4,
        "span": {"message_index": 3, "start": 424, "end": 570},
        "context": None,
    }
    assert long_score["value"] == pytest.approx(0.7)

    # `context` repeats the system prompt in every chunk as a span of its own, and
    # `overlap` starts each later range of the page inside the range before it.
    repeating = rule.replace("split(messages, 40)", "split(messages, 40, 12, system_messages)")
    repeated = await plugin.scan(rule_text=repeating, conversation=LONG_TOOL)
    assert repeated["matched"] is True
    request, *pieces = repeated["decision_path"]["path"][1]["score"]["chunks"]
    system = {"message_index": 0, "start": 0, "end": len(SYSTEM["content"])}
    assert all(chunk["context"] == system for chunk in (request, *pieces))
    assert request["span"] == {"message_index": 1, "start": 0, "end_message_index": 2, "end": 0}

    def bounds(span: dict) -> tuple[tuple[int, int], tuple[int, int]]:
        last = span.get("end_message_index", span["message_index"])
        return (span["message_index"], span["start"]), (last, span["end"])

    ranges = [bounds(piece["span"]) for piece in pieces]
    # The first page range uses its full budget, so it cannot repeat the request's
    # text-less tool call. Later page ranges have room for overlap.
    assert ranges[0][0] == (3, 0)
    assert ranges[-1][1] == (3, len(PAGE))
    assert all(
        left[0] < right[0] < left[1] < right[1]
        for left, right in zip(ranges, ranges[1:], strict=False)
    )
    assert [piece["value"] for piece in pieces] == [
        pytest.approx(_keyword_score(_chunk_text(LONG_TOOL, piece))) for piece in pieces
    ]

    # A context that selects nothing, here `system_messages` with no system message,
    # repeats nothing.
    unprompted = await plugin.scan(
        rule_text=repeating, conversation={"messages": LONG_TOOL["messages"][1:]}
    )
    assert unprompted["completion_state"] == "complete"
    unprompted_chunks = unprompted["decision_path"]["path"][1]["score"]["chunks"]
    assert unprompted_chunks and all(chunk["context"] is None for chunk in unprompted_chunks)

    # A view keeps transcript order, so a context that is not leading fails the scan.
    trailing = rule.replace("split(messages, 40)", "split(messages, 40, messages[-1:])")
    misplaced = await plugin.scan(rule_text=trailing, conversation=LONG_TOOL)
    assert misplaced["completion_state"] != "complete"
    assert "context must come before" in json.dumps(misplaced["diagnostics"])

    # An input with gaps packs across them: the fiction's two user turns form one chunk,
    # one span from the first user message to the last; the turns between are not read.
    grouping = rule.replace("split(messages, 40)", "split(user_messages, 40)")
    grouped = await plugin.scan(rule_text=grouping, conversation=FICTION)
    (together,) = grouped["decision_path"]["path"][1]["score"]["chunks"]
    assert together["span"] == _whole(FICTION, 1, 5)
    assert together["value"] == pytest.approx(
        _keyword_score(FICTION["messages"][1]["content"] + FICTION["messages"][5]["content"])
    )

    # Tools are not messages, so split takes none; the chunk's score may pass them.
    with_tools = plugin.compile(
        rule_text=rule.replace("split(messages, 40)", "split(messages, 40, tools)")
    )
    assert with_tools["valid"] is False
    assert "takes no tools" in with_tools["diagnostics"][0]["message"]

    # The benign story reaches the score node and misses; its score is the closest miss.
    assert [entry["outcome"] for entry in benign["decision_path"]["path"]] == [True, False]
    assert benign["decision_path"]["path"][1]["score"]["value"] == pytest.approx(0.1)
    assert benign["decision_path"]["path"][1]["score"]["matching"] == 0

    # Without a story the score node is skipped; exploring scores it anyway.
    assert no_story["decision_path"]["path"] == [{"node": story_id, "outcome": False}]
    exploration = no_story["decision_path"]["exploration"]
    assert exploration["selected"] is True and exploration["failed"] == []
    (off,) = exploration["off_path"]
    assert off["node"] == score_id
    assert off["score"]["value"] == pytest.approx(
        _keyword_score(NO_STORY["messages"][1]["content"])
    )
    assert "off_path" not in long_tool["decision_path"]["exploration"]

    # Telemetry keeps each step's position, outcome, value and threshold, and the one span
    # a score came from, never the chunk list, so its size does not grow with the input.
    long_path = long_tool["decision_path"]
    long_score = long_path["path"][1]["score"]
    order = sorted(tree["nodes"])
    story_at, score_at = order.index(story_id), order.index(score_id)
    assert telemetry_record(tree, long_path) == {
        "tree_sha256": tree["tree_sha256"],
        "config_sha256": long_path["config_sha256"],
        "parameter_digest": long_path["parameter_digest"],
        "conversation_sha256": long_path["conversation_sha256"],
        "leaf": "match",
        "steps": [
            {"node": story_at, "outcome": True},
            {
                "node": score_at,
                "outcome": True,
                "value": long_score["value"],
                "threshold": 0.6,
                "span": {"message_index": 3, "start": 424, "end": 570},
            },
        ],
    }
    explored = telemetry_record(tree, no_story["decision_path"])
    assert explored["steps"] == [{"node": story_at, "outcome": False}]
    assert explored["failed"] == []
    ((off_step),) = explored["explored"]
    assert (off_step["node"], off_step["value"]) == (score_at, off["score"]["value"])
    # A copy of the tree with its nodes in another order has the same hash, and the
    # record names the same steps by the same positions.
    reordered = {**tree, "nodes": dict(reversed(list(tree["nodes"].items())))}
    assert telemetry_record(reordered, long_path) == telemetry_record(tree, long_path)
    assert len(json.dumps(telemetry_record(tree, long_path))) < len(json.dumps(long_path)) / 2
    with pytest.raises(ValueError, match="walked tree"):
        telemetry_record(tree, {**long_path, "tree_sha256": "0" * 64})

    # A skipped check whose model fails is listed under `failed`; the verdict stands.
    async def unavailable(config, messages, *, timeout=None):
        raise LLMError("judge unavailable", {"code": "unavailable"})

    judge.score = unavailable
    # A conversation the server has not scored: it reuses the scores it has.
    unscored = {"messages": [SYSTEM, {"role": "user", "content": "How do I stop a hung service?"}]}
    failing = await plugin.scan(rule_text=rule, conversation=unscored, explore_rate=1.0)
    assert failing["matched"] is False and failing["completion_state"] == "complete"
    assert failing["decision_path"]["exploration"]["off_path"] == []
    assert failing["decision_path"]["exploration"]["failed"] == [score_id]
    del judge.score

    # Without explain, a path keeps only the deciding chunk.
    brief = await plugin.scan(rule_text=rule, conversation=FICTION, explain=False)
    assert brief["decision_path"]["path"][1]["score"] == {
        key: value for key, value in decision.items() if key != "chunks"
    }

    # Node IDs survive a rename and a metadata edit. A threshold change gives the score
    # node a new ID but keeps its score key, so stored scores answer the new threshold.
    renamed = plugin.decision_tree(rule_text=_rule(0.6, name="Renamed"))["decision_tree"]["Renamed"]
    assert set(renamed["nodes"]) == set(tree["nodes"])
    (stricter,) = plugin.decision_tree(rule_text=_rule(0.9))["decision_tree"].values()
    stricter_score = stricter["nodes"][stricter["nodes"][stricter["root"]]["true"]]
    assert stricter["root"] == story_id
    assert stricter_score["id"] != score_id
    assert stricter_score["score_key"] == score["score_key"]


@pytest.mark.asyncio
async def test_flows_count_scans_through_the_tree_and_move_its_thresholds(plugin):
    plugin, _ = plugin
    rule = """rule "Long fiction"
parameters:
    cut: number
condition:
    len(messages) > 4
    and (any chunk in model("judge").split(messages, 40) where
        model("judge").score(chunk, "fiction") > params.cut)
"""
    # A size check is a number node, tested first; the score's bound is its parameter.
    (tree,) = plugin.decision_tree(rule_text=rule)["decision_tree"].values()
    size_id = tree["root"]
    size = tree["nodes"][size_id]
    score_id = size["true"]
    assert size["kind"] == "number" and size["false"] == "no_match"
    assert size["number_form"] == {"operator": ">", "threshold": 4.0}
    assert tree["nodes"][score_id]["score_form"]["threshold"] == {"parameter": "cut"}

    batch = [FICTION, BENIGN_STORY, NO_STORY, LONG_TOOL]
    result = await plugin.scan(rule_text=rule, conversations=batch, parameters={"cut": 0.6})
    paths = [item["decision_path"] for item in result["results"]]
    # Every path names its tree and records the size it compared, true or false.
    assert all(path["tree_sha256"] == tree["tree_sha256"] for path in paths)
    assert [path["path"][0]["number"] for path in paths] == [
        {"value": len(conversation["messages"]), "operator": ">", "threshold": 4.0}
        for conversation in batch
    ]
    assert [path["leaf"] for path in paths] == ["match", "no_match", "no_match", "no_match"]

    labels = ["exempt", "exempt", "unsafe", "unsafe"]

    async def flows(**arguments):
        counted = (
            await mcp.call_tool(
                "quarry_decision_flows",
                {"tree": tree, "paths": paths, "labels": labels, **arguments},
            )
        ).structured_content
        return counted

    as_scanned = await flows()
    assert as_scanned["steps"] == {
        size_id: {"true": {"exempt": 2}, "false": {"unsafe": 2}},
        score_id: {"true": {"exempt": 1}, "false": {"exempt": 1}},
    }
    assert as_scanned["leaves"] == {
        "match": {"exempt": 1},
        "no_match": {"exempt": 1, "unsafe": 2},
        "unknown": {},
        "error": {},
    }
    # Moving the parameter re-compares the recorded cat-story score (0.1) without scanning.
    lower_cut = await flows(thresholds={"cut": 0.05})
    assert lower_cut["thresholds"] == {score_id: 0.05}
    assert lower_cut["leaves"]["match"] == {"exempt": 2}
    # A lower size bound sends the short conversations to a score they never ran.
    shorter = await flows(thresholds={size_id: 1})
    assert shorter["steps"][size_id] == {"true": {"exempt": 2, "unsafe": 2}, "false": {}}
    assert shorter["leaves"]["unknown"] == {"unsafe": 2}

    # A scan that failed at the score has an unanswered check on its own route.
    failed = {**paths[1], "leaf": None, "path": paths[1]["path"][:1]}
    counted = (
        await mcp.call_tool("quarry_decision_flows", {"tree": tree, "paths": [failed]})
    ).structured_content
    assert counted["leaves"]["error"] == {"all": 1}

    # Paths map onto one tree only; a threshold must name a score or number node.
    other = [{**paths[0], "tree_sha256": "0" * 64}]
    with pytest.raises(ToolError, match="walked tree"):
        await mcp.call_tool("quarry_decision_flows", {"tree": tree, "paths": other})
    with pytest.raises(ToolError, match="name no score or number node"):
        await flows(thresholds={"missing": 1})
    # A tree must be the one its hash names, and a walk cannot loop.
    edited = {**tree, "root": score_id}
    with pytest.raises(ToolError, match="not the hash of its content"):
        await mcp.call_tool("quarry_decision_flows", {"tree": edited, "paths": paths})
    nodes = {**tree["nodes"], size_id: {**size, "true": size_id}}
    looping = {**tree, "nodes": nodes}
    looping["tree_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in looping.items() if key != "tree_sha256"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    looped = [{**paths[0], "tree_sha256": looping["tree_sha256"]}]
    with pytest.raises(ToolError, match="edges loop"):
        await mcp.call_tool("quarry_decision_flows", {"tree": looping, "paths": looped})


@pytest.mark.asyncio
async def test_exploration_leaves_the_verdict_and_its_evidence_as_the_path_left_them(plugin):
    # The story check decides the match. Exploration also runs the skipped flow check,
    # which holds too, but its messages must not become the match's evidence.
    plugin, _ = plugin
    rule = """rule "Story or poison then villain"
condition:
    (any m in user_messages where m.content contains "story")
    or ((any r in tool_responses where r.content contains "Arsenic")
        before (any m in user_messages where m.content contains "villain"))
"""
    plain = await plugin.scan(rule_text=rule, conversation=FICTION, visualize=True)
    explored = await plugin.scan(
        rule_text=rule, conversation=FICTION, visualize=True, explore_rate=1.0
    )

    (off,) = explored["decision_path"]["exploration"]["off_path"]
    assert off["outcome"] is True
    assert explored["matched"] is plain["matched"] is True

    def without_path(payload):
        return {
            key: value for key, value in payload.items() if key not in {"decision_path", "timings"}
        }

    assert without_path(explored) == without_path(plain)


@pytest.mark.asyncio
async def test_an_equality_score_is_a_score_node_that_records_its_score(plugin):
    # `score(...) = 1` is how a judge without logprobs is written; its score is still
    # recorded on both outcomes, and its threshold can be moved like any other.
    plugin, _ = plugin
    rule = """rule "Judged story"
condition:
    (any m in user_messages where m.content contains "story")
    and model("judge").score(messages, "fiction") == 1
"""
    (tree,) = plugin.decision_tree(rule_text=rule)["decision_tree"].values()
    score_id = tree["nodes"][tree["root"]]["true"]
    score = tree["nodes"][score_id]
    assert score["kind"] == "score"
    assert score["score_form"] == {"quantifier": "none", "operator": "=", "threshold": 1.0}

    result = await plugin.scan(rule_text=rule, conversations=[FICTION, BENIGN_STORY])
    recorded = [item["decision_path"]["path"][1] for item in result["results"]]
    assert [(entry["outcome"], entry["score"]["value"]) for entry in recorded] == [
        (True, 1.0),
        (False, pytest.approx(0.1)),
    ]
    assert all(entry["score"]["threshold"] == 1.0 for entry in recorded)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "comparison",
    [
        'model("judge").score(messages, "fiction") > 0.5',
        '0.5 < model("judge").score(messages, "fiction")',
    ],
    ids=["score-on-the-left", "score-on-the-right"],
)
async def test_a_score_node_records_the_score_against_its_threshold_either_way(plugin, comparison):
    plugin, _ = plugin
    rule = f"""rule "Judged story"
condition:
    (any m in user_messages where m.content contains "story")
    and {comparison}
"""

    result = await plugin.scan(rule_text=rule, conversations=[FICTION])

    (item,) = result["results"]
    score = item["decision_path"]["path"][1]["score"]
    assert (score["value"], score["operator"], score["threshold"]) == (1.0, ">", 0.5)


@pytest.mark.asyncio
async def test_flow_nodes_decompose_and_captures_keep_one_node(plugin):
    plugin, _ = plugin
    flow = """rule "Injected then echoed"
strings:
    $inject = "ignore (all )?previous instructions" regex nocase
condition:
    ((any tr in tool_responses where $inject in tr.content)
     before(within: 3)
     (any msg in assistant_messages where msg.content contains "ignore"))
    and model("judge").score(messages[-2:], "injection") > 0.3
"""
    (tree,) = plugin.decision_tree(rule_text=flow)["decision_tree"].values()
    kinds = [node["kind"] for node in tree["nodes"].values()]
    assert sorted(kinds) == ["check", "score"]
    assert tree["nodes"][tree["root"]]["kind"] == "check"

    capture = """rule "Email reused"
strings:
    $email = "[a-z0-9._%+-]+@[a-z0-9.-]+" regex
condition:
    (any msg in user_messages where $email in msg.content as user_email)
    before
    (any tc in tool_calls where tc.name == "send_email" and user_email in tc.arguments.*)
"""
    (tree,) = plugin.decision_tree(rule_text=capture)["decision_tree"].values()
    assert list(tree["nodes"]) == [tree["root"]]
    result = await plugin.scan(rule_text=capture, conversation=FICTION)
    assert result["decision_path"]["path"] == [{"node": tree["root"], "outcome": False}]
    assert result["decision_path"]["leaf"] == "no_match"


@pytest.mark.asyncio
async def test_the_decision_tree_tool_reports_the_compiled_policys_tree():
    # One tree, three ways to get it: the MCP tool, the compiled policy, and that policy
    # loaded from bytes. All three agree, so node IDs mean the same thing everywhere.
    rule = (
        'rule "Story without sending"\ncondition:\n'
        '    (any m in user_messages where m.content contains "story")\n'
        '    and not called_tool("send_email")\n'
    )
    reported = (await mcp.call_tool("quarry_decision_tree", {"rule_text": rule})).structured_content
    policy = compile_policy(rule)

    assert reported["valid"] is True
    (tree,) = reported["decision_tree"].values()
    assert tree == policy.decision_tree()
    assert load_compiled_policy(policy.to_bytes()).decision_tree() == tree
    root = tree["nodes"][tree["root"]]
    assert root["label"] == 'any m in user_messages where (m.content contains "story")'
    assert (root["false"], tree["nodes"][root["true"]]["false"]) == ("no_match", "match")

    broken = (
        await mcp.call_tool(
            "quarry_decision_tree", {"rule_text": 'rule "Broken"\ncondition:\n    called_tool(\n'}
        )
    ).structured_content
    assert broken["valid"] is False and broken["diagnostics"]
    assert "decision_tree" not in broken


@pytest.mark.asyncio
async def test_where_a_model_is_served_changes_a_paths_config_hash_but_not_its_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Node IDs name what a check asks; config_sha256 names what answered it, where the
    # model and its tokenizer are served included.
    monkeypatch.setattr(
        "quarry.integrations.model_inference._default_adapters",
        lambda: {"judge": _KeywordJudge(), "words": _WordTokens()},
    )

    def served(model: str, endpoint: str, tokenizer: str) -> QuarryPlugin:
        config = QuarryConfig.from_mapping(
            {
                "config_version": 2,
                "llm_rate_limits": {
                    "default": {"requests_per_interval": None, "max_concurrent": 8}
                },
                "models": {
                    "judge": {
                        "adapter": "judge",
                        "model": model,
                        "endpoint": endpoint,
                        "region": "us-west-2",
                        "tokens": {
                            "adapter": "words",
                            "endpoint": tokenizer,
                            "extraction": {"count": "count"},
                        },
                        "score": {
                            "message_builder": "constitution",
                            "boundary_seed": 7,
                            "tokens": {"negative": "0", "positive": "1"},
                        },
                    }
                },
            },
            tmp_path,
            require_rules=False,
        )
        return QuarryPlugin(config)

    async def path(plugin: QuarryPlugin) -> dict:
        result = await plugin.scan(rule_text=_rule(0.6), conversation=FICTION)
        return result["decision_path"]

    here = await path(served("keyword-judge", "https://a.example/v1", "https://a.example/t"))
    moved = await path(served("keyword-judge", "https://b.example/v1", "https://b.example/t"))
    other = await path(served("keyword-judge-2", "https://a.example/v1", "https://a.example/t"))

    assert [step["node"] for step in moved["path"]] == [step["node"] for step in here["path"]]
    assert moved["tree_sha256"] == here["tree_sha256"]
    assert moved["config_sha256"] != here["config_sha256"]
    assert other["path"][1]["node"] != here["path"][1]["node"]
