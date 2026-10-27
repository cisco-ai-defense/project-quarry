# Copyright 2026 Cisco Systems, Inc. and its affilitates
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

"""Generated rules and conversations, each rule paired with a direct reading of its meaning.

A generated ``Rule`` carries its DSL text and ``holds(env, conversation)``: a deliberately
naive first-order evaluator of that text over the conversation, written without reference to
the compiler. The property tests compare the scanner's verdict with it.

A conversation is the scanner's input dict: ``messages`` (user, assistant, system and
developer turns, assistant turns with tool calls, tool results) and ``tools`` (tool
definitions).
"""

from __future__ import annotations

import json
import operator
import re
from collections.abc import Callable
from dataclasses import dataclass

from hypothesis import strategies as st

_WORDS = ["a", "b", "ab", "go"]
_SURFACES = {
    "user_messages": "user",
    "assistant_messages": "assistant",
    "tool_calls": "call",
    "tool_responses": "tool",
}
_TOOL_NAMES = ["x", "y", "z"]


def _timeline(conversation: dict) -> list[dict]:
    """Timeline events in order: each message, then each of its tool calls. A tool result is
    one event, named after the call it answers."""
    events = []
    names: dict[str, str] = {}
    for message in conversation["messages"]:
        if message["role"] == "tool":
            name = names[message["tool_call_id"]]
            events.append({"kind": "tool", "name": name, "content": message["content"]})
            continue
        # `content` is flattened text: a message without text (a tool-call turn) reads "".
        events.append({"kind": message["role"], "content": message["content"] or ""})
        for call in message.get("tool_calls") or []:
            arguments = json.loads(call["function"]["arguments"])
            names[call["id"]] = call["function"]["name"]
            events.append(
                {"kind": "call", "name": call["function"]["name"], "content": arguments["cmd"]}
            )
    for position, entry in enumerate(events):
        entry["pos"] = position
    return events


def _elements(conversation: dict, kind: str) -> list[dict]:
    return [entry for entry in _timeline(conversation) if entry["kind"] == kind]


def _tool_definitions(conversation: dict) -> list[dict]:
    """The tool definitions, each read like an element: its name, and its description as text."""
    return [
        {
            "kind": "tooldef",
            "name": tool["function"]["name"],
            "content": tool["function"]["description"],
        }
        for tool in conversation.get("tools") or []
    ]


def _text(element: dict) -> str:
    return element["content"]


_MESSAGE_KINDS = ("user", "assistant", "system", "developer", "tool")


@dataclass(frozen=True)
class _Collection:
    """A comprehension collection: its text, its elements' kind, and its elements."""

    text: str
    kind: str
    members: Callable[[dict], list[dict]]


_TOOLS = _Collection("tools", "tooldef", _tool_definitions)


@st.composite
def _collections(draw) -> _Collection:
    choice = draw(st.sampled_from(["surface", "surface", "messages", "accessor"]))
    if choice == "accessor":
        name = draw(st.sampled_from(["x", "y"]))
        if draw(st.booleans()):
            return _Collection(
                f'get_tool_calls("{name}")',
                "call",
                lambda conv: [e for e in _elements(conv, "call") if e["name"] == name],
            )
        return _Collection(
            f'get_tool_responses("{name}")',
            "tool",
            lambda conv: [e for e in _elements(conv, "tool") if e["name"] == name],
        )
    if choice == "messages":
        surface, kind = "messages", "message"
        members = lambda conv: [e for e in _timeline(conv) if e["kind"] in _MESSAGE_KINDS]  # noqa: E731
    else:
        surface = draw(st.sampled_from(sorted(_SURFACES)))
        kind = _SURFACES[surface]
        members = lambda conv, kind=kind: _elements(conv, kind)  # noqa: E731
    if draw(st.integers(min_value=0, max_value=3)) == 0:
        # A window resolves like a Python slice of the surface.
        start = draw(st.one_of(st.none(), st.integers(min_value=-3, max_value=3)))
        stop = draw(st.one_of(st.none(), st.integers(min_value=-3, max_value=3)))
        bounds = f"{'' if start is None else start}:{'' if stop is None else stop}"
        return _Collection(
            f"{surface}[{bounds}]", kind, lambda conv, m=members: m(conv)[start:stop]
        )
    return _Collection(surface, kind, members)


@dataclass(frozen=True)
class Rule:
    """Rule text plus its meaning, evaluated under an assignment of loop variables."""

    text: str
    holds: Callable[[dict, dict], bool]
    # `strings:` definitions the condition refers to (identical names define identical patterns).
    strings: tuple[str, ...] = ()

    def source(self) -> str:
        block = "strings:\n" + "".join(f"    {d}\n" for d in self.strings) if self.strings else ""
        return f'rule "R"\n{block}condition:\n    {self.text}\n'


def _merged(*rules) -> tuple[str, ...]:
    return tuple(dict.fromkeys(d for rule in rules for d in rule.strings))


# Declared patterns: name -> (pattern, modifier). Names are fixed so equal names are equal.
_PATTERNS = {
    **{
        f"${word}_{mod or 'plain'}": (word, mod)
        for word in ("a", "b", "go")
        for mod in (None, "nocase", "fullword")
    },
    "$w_regex": ("a|go", "regex"),
}


def _pattern_matches(name: str, text: str) -> list[str]:
    """Non-overlapping matched texts of a declared pattern, as the reference defines them."""
    pattern, mod = _PATTERNS[name]
    expression = {
        "nocase": f"(?i){re.escape(pattern)}",
        "fullword": rf"\b{re.escape(pattern)}\b",
        "regex": pattern,
    }.get(mod, re.escape(pattern))
    return [match.group(0) for match in re.finditer(expression, text)]


def _pattern_hits(name: str, text: str) -> int:
    return len(_pattern_matches(name, text))


def _definition(name: str) -> str:
    pattern, mod = _PATTERNS[name]
    return f'{name} = "{pattern}"' + (f" {mod}" if mod else "")


# Regexes whose Python `re.search` and RE2 meanings coincide on the fuzzer's ASCII texts.
_REGEXES = ["^a", "b$", "a.", "^go$", "(a|go)b?"]


_COMPARE = {
    ">=": operator.ge,
    ">": operator.gt,
    "<=": operator.le,
    "<": operator.lt,
    "==": operator.eq,
    "!=": operator.ne,
}


def _field(var: str, kind: str) -> str:
    if kind == "call":
        return f"{var}.arguments.cmd"
    if kind == "tooldef":
        return f"{var}.description"
    return f"{var}.content"


def _timeline_scope(scope: tuple[tuple[str, str], ...]) -> tuple[tuple[str, str], ...]:
    """The loop variables bound to timeline events: a tool definition has no position."""
    return tuple((var, kind) for var, kind in scope if kind != "tooldef")


@st.composite
def _rules(
    draw,
    scope: tuple[tuple[str, str], ...] = (),
    depth: int = 3,
    size: int = 4,
    choice: str | None = None,
    captured: bool = False,
) -> Rule:
    """A rule over the loop variables in ``scope``; ``captured`` lets element conditions test
    the captured value ``v``."""
    options = ["fact"]
    if scope:
        options += ["element"] * 3
    if len(scope) >= 2:
        options += ["join"]
    if size > 0:
        options += ["and", "or", "not", "flow"]
        if depth > 0:
            options += ["any", "all"]
            if all(kind != "tooldef" for _, kind in scope):
                options += ["tools"]
    if choice is None:
        choice = draw(st.sampled_from(options))

    if choice == "fact":
        shape = draw(st.sampled_from(["called", "called", "said", "said", "size", "tool"]))
        if shape == "tool":
            name = draw(st.sampled_from(_TOOL_NAMES))
            if draw(st.booleans()):
                return Rule(
                    f'has_tool("{name}")',
                    lambda env, conv: any(t["name"] == name for t in _tool_definitions(conv)),
                )
            op = draw(st.sampled_from(sorted(_COMPARE)))
            bound = draw(st.sampled_from([0, 1, 2]))
            compare = _COMPARE[op]
            return Rule(
                f"len(tools) {op} {bound}",
                lambda env, conv: compare(len(_tool_definitions(conv)), bound),
            )
        if shape == "size":
            op = draw(st.sampled_from(sorted(_COMPARE)))
            bound = draw(st.sampled_from([0, 1, 2, 3, 0.5, 1.5]))
            compare = _COMPARE[op]
            size_of = draw(st.sampled_from(["calls", "len", "filter"]))
            if size_of == "calls":
                name = draw(st.sampled_from(["x", "y"]))
                return Rule(
                    f'count_tool_calls("{name}") {op} {bound}',
                    lambda env, conv: compare(
                        sum(c["name"] == name for c in _elements(conv, "call")), bound
                    ),
                )
            collection = draw(_collections())
            if size_of == "len":
                return Rule(
                    f"len({collection.text}) {op} {bound}",
                    lambda env, conv: compare(len(collection.members(conv)), bound),
                )
            var = f"f{len(scope)}"
            keep = draw(_rules(((var, collection.kind),), depth=0, size=0, choice="element"))
            return Rule(
                f"count([{var} for {var} in {collection.text} if {keep.text}]) {op} {bound}",
                lambda env, conv: compare(
                    sum(keep.holds({var: e}, conv) for e in collection.members(conv)), bound
                ),
                keep.strings,
            )
        if shape == "called":
            name = draw(st.sampled_from(["x", "y"]))
            return Rule(
                f'called_tool("{name}")',
                lambda env, conv: any(c["name"] == name for c in _elements(conv, "call")),
            )
        word = draw(st.sampled_from(_WORDS))
        return Rule(
            f'user_said("{word}")',
            lambda env, conv: any(word in u["content"] for u in _elements(conv, "user")),
        )
    if choice == "element":
        var, kind = draw(st.sampled_from(scope))
        if kind in ("call", "tool", "tooldef") and draw(st.booleans()):
            name = draw(st.sampled_from(_TOOL_NAMES))
            return Rule(f'{var}.name == "{name}"', lambda env, conv: env[var]["name"] == name)
        field = _field(var, kind)
        forms = ["contains", "contains", "in", "count", "set", "equals", "regex"]
        forms += ["length", "occurs", "differs", "block"]
        if kind == "tooldef":
            # A tool definition is tested on its text fields, not counted or measured.
            forms = ["contains", "contains", "in", "equals", "regex", "differs"]
        form = draw(st.sampled_from(forms))
        if captured and draw(st.booleans()):
            return Rule(f"{field} contains v", lambda env, conv: env["v"] in _text(env[var]))
        if form in ("length", "occurs"):
            op = draw(st.sampled_from(sorted(_COMPARE)))
            bound = draw(st.sampled_from([0, 1, 2, 3, 0.5, 1.5]))
            compare = _COMPARE[op]
            if form == "length":
                return Rule(
                    f"len({field}) {op} {bound}",
                    lambda env, conv: compare(len(_text(env[var])), bound),
                )
            word = draw(st.sampled_from(_WORDS))
            return Rule(
                f'count({field}, "{word}") {op} {bound}',
                lambda env, conv: compare(_text(env[var]).count(word), bound),
            )
        if form == "block" and kind != "call":
            # A pattern block names the element itself, not one of its fields.
            word = draw(st.sampled_from(_WORDS))
            return Rule(
                f'{var} matches {{ content == "{word}" }}',
                lambda env, conv: _text(env[var]) == word,
            )
        if form == "differs":
            word = draw(st.sampled_from([*_WORDS, ""]))
            return Rule(f'{field} != "{word}"', lambda env, conv: _text(env[var]) != word)
        if form == "in":
            name = draw(st.sampled_from(sorted(_PATTERNS)))
            return Rule(
                f"{name} in {field}",
                lambda env, conv: _pattern_hits(name, _text(env[var])) > 0,
                (_definition(name),),
            )
        if form == "count":
            name = draw(st.sampled_from(sorted(_PATTERNS)))
            least = draw(st.integers(min_value=1, max_value=2))
            return Rule(
                f"#{name[1:]} in {field} >= {least}",
                lambda env, conv: _pattern_hits(name, _text(env[var])) >= least,
                (_definition(name),),
            )
        if form == "set":
            names = draw(
                st.lists(st.sampled_from(sorted(_PATTERNS)), min_size=2, max_size=3, unique=True)
            )
            quantifier = draw(st.sampled_from(["any", "all", "1", "2"]))
            need = {"any": 1, "all": len(names)}.get(quantifier) or int(quantifier)
            return Rule(
                f"{quantifier} of ({', '.join(names)}) in {field}",
                lambda env, conv: sum(_pattern_hits(n, _text(env[var])) > 0 for n in names) >= need,
                tuple(_definition(n) for n in names),
            )
        if form == "equals":
            word = draw(st.sampled_from([*_WORDS, ""]))
            return Rule(f'{field} == "{word}"', lambda env, conv: _text(env[var]) == word)
        if form == "regex":
            regex = draw(st.sampled_from(_REGEXES))
            return Rule(
                f'{field} ~ "{regex}"',
                lambda env, conv: re.search(regex, _text(env[var])) is not None,
            )
        word = draw(st.sampled_from(_WORDS))
        return Rule(
            f'{field} contains "{word}"',
            lambda env, conv: word in _text(env[var]),
        )
    if choice == "join":
        (outer, outer_kind), (inner, inner_kind) = draw(st.permutations(scope))[:2]
        return Rule(
            f"{_field(outer, outer_kind)} contains {_field(inner, inner_kind)}",
            lambda env, conv: bool(_text(env[inner])) and _text(env[inner]) in _text(env[outer]),
        )
    if choice == "flow":
        # Flow operands are timeline events; a tool definition in scope takes no part.
        flow_scope = _timeline_scope(scope)
        later = draw(_flow_operands(flow_scope, captured=captured))
        earlier = draw(_flow_operands(flow_scope, captured=captured))
        if draw(st.integers(min_value=0, max_value=3)) == 0:
            sustained, release = later, earlier

            def holds(env, conv, sustained=sustained, release=release):
                if sustained.names or release.names:
                    return False  # `until` has no per-name form; the rule is rejected at load
                # The sustained side holds at every event of its kind before the release.
                held = sustained.positions(env, conv)
                return any(
                    all(i in held for i in sustained.domain(env, conv) if i < k)
                    for k in release.positions(env, conv)
                )

            # A sustained comprehension reads the same with either quantifier: every element
            # before the release satisfies its body.
            text = sustained.text
            if text.startswith("(any ") and draw(st.booleans()):
                text = "(all " + text[len("(any ") :]
            return Rule(f"({text} until {release.text})", holds, _merged(sustained, release))
        within = draw(st.one_of(st.none(), st.integers(min_value=1, max_value=3)))
        # `A after B` is `B before A`, bounded or not, and `right_after` is `after(within: 1)`.
        spelled_before = draw(st.booleans())
        if within == 1 and draw(st.booleans()):
            word = "right_before" if spelled_before else "right_after"
        else:
            word = "before" if spelled_before else "after"
            if within is not None:
                word += f"(within: {within})"

        def holds(env, conv, later=later, earlier=earlier, within=within):
            def pair(late, early) -> bool:
                starts, ends = early.positions(env, conv), late.positions(env, conv)
                return any(
                    i < j and (within is None or j - i <= within) for i in starts for j in ends
                )

            return all(pair(late, early) for late in later.each() for early in earlier.each())

        strings = _merged(later, earlier)
        if spelled_before:
            return Rule(f"({earlier.text} {word} {later.text})", holds, strings)
        return Rule(f"({later.text} {word} {earlier.text})", holds, strings)
    if choice == "not":
        operand = draw(_rules(scope, depth, size - 1, captured=captured))
        return Rule(
            f"(not {operand.text})",
            lambda env, conv: not operand.holds(env, conv),
            operand.strings,
        )
    if choice in ("and", "or"):
        left = draw(_rules(scope, depth, size - 1, captured=captured))
        right = draw(_rules(scope, depth, size - 1, captured=captured))
        combine = all if choice == "and" else any
        return Rule(
            f"({left.text} {choice} {right.text})",
            lambda env, conv: combine(part.holds(env, conv) for part in (left, right)),
            _merged(left, right),
        )
    if choice == "tools":
        collection = _TOOLS
        choice = draw(st.sampled_from(["any", "all"]))
    else:
        collection = draw(_collections())
    var = f"v{len(scope)}"
    body = draw(_rules((*scope, (var, collection.kind)), depth - 1, size - 1, captured=captured))
    quantify = any if choice == "any" else all
    return Rule(
        f"({choice} {var} in {collection.text} where {body.text})",
        lambda env, conv: quantify(
            body.holds({**env, var: element}, conv) for element in collection.members(conv)
        ),
        body.strings,
    )


@dataclass(frozen=True)
class _Operand:
    """A flow operand: its text and the timeline positions where it holds."""

    text: str
    positions: Callable[[dict, dict], set[int]]
    # The events of the operand's kind: where a sustained `until` side must hold.
    domain: Callable[[dict, dict], set[int]]
    # A multi-name `called_tools(...)` operand is one flow per name.
    names: tuple[str, ...] = ()
    strings: tuple[str, ...] = ()

    def each(self) -> list[_Operand]:
        return [_called(name) for name in self.names] if self.names else [self]


def _called(name: str) -> _Operand:
    return _Operand(
        f'called_tool("{name}")',
        lambda env, conv: {e["pos"] for e in _elements(conv, "call") if e["name"] == name},
        lambda env, conv: {e["pos"] for e in _elements(conv, "call")},
    )


@dataclass(frozen=True)
class _Needle:
    """The text a `contains` looks for: a literal word, or the captured value `v`."""

    text: str
    value: Callable[[dict], str]


@st.composite
def _needles(draw, captured: bool) -> _Needle:
    if captured and draw(st.booleans()):
        return _Needle("v", lambda env: env["v"])
    word = draw(st.sampled_from(_WORDS))
    return _Needle(f'"{word}"', lambda env: word)


@st.composite
def _operand_facts(draw, captured: bool) -> Rule | None:
    """A condition that does not read the operand's element: none, a call, or (with a capture)
    another collection holding the captured value."""
    shape = draw(st.sampled_from(["none", "call", *["value"] * captured]))
    if shape == "none":
        return None
    if shape == "call":
        name = draw(st.sampled_from(["x", "y"]))
        return Rule(
            f'called_tool("{name}")',
            lambda env, conv: any(c["name"] == name for c in _elements(conv, "call")),
        )
    collection = draw(_collections())
    return Rule(
        f"(any a in {collection.text} where {_field('a', collection.kind)} contains v)",
        lambda env, conv: any(env["v"] in _text(e) for e in collection.members(conv)),
    )


@st.composite
def _flow_operands(
    draw, scope: tuple[tuple[str, str], ...], size: int = 1, captured: bool = False
) -> _Operand:
    if size > 0 and draw(st.integers(min_value=0, max_value=3)) == 0:
        # An `or` operand holds wherever either alternative does.
        left = draw(_flow_operands(scope, size - 1, captured))
        right = draw(_flow_operands(scope, size - 1, captured))
        return _Operand(
            f"({left.text} or {right.text})",
            lambda env, conv: left.positions(env, conv) | right.positions(env, conv),
            lambda env, conv: left.domain(env, conv) | right.domain(env, conv),
            strings=_merged(left, right),
        )
    if draw(st.integers(min_value=0, max_value=3)) == 0:
        # A comprehension operand holds at its matching elements' events; a conversation-wide
        # fact in its body has one truth value for the whole conversation.
        collection = draw(_collections())
        kind = collection.kind
        var = f"o{len(scope)}"
        members = collection.members
        if draw(st.booleans()):
            # Any body: conditions on the element, joins to enclosing elements, conversation-wide
            # facts and nested comprehensions, each part read at the element's event.
            body_rule = draw(_rules((*scope, (var, kind)), depth=1, size=2, captured=captured))
            return _Operand(
                f"(any {var} in {collection.text} where {body_rule.text})",
                lambda env, conv: {
                    e["pos"] for e in members(conv) if body_rule.holds({**env, var: e}, conv)
                },
                lambda env, conv: {e["pos"] for e in members(conv)},
                strings=body_rule.strings,
            )
        needle = draw(_needles(captured))
        fact = draw(_operand_facts(captured))
        negated = draw(st.booleans())
        body = f"{_field(var, kind)} contains {needle.text}"
        if negated:
            body = f"not ({body})"
        if fact is not None:
            body += f" and {fact.text}"

        def positions(env, conv, collection=collection, needle=needle, fact=fact, negated=negated):
            if fact is not None and not fact.holds(env, conv):
                return set()
            value = needle.value(env)
            return {
                e["pos"] for e in collection.members(conv) if (value in e["content"]) != negated
            }

        return _Operand(
            f"(any {var} in {collection.text} where {body})",
            positions,
            lambda env, conv, collection=collection: {e["pos"] for e in collection.members(conv)},
        )
    if scope and draw(st.booleans()):
        var, kind = draw(st.sampled_from(scope))
        needle = draw(_needles(captured))
        return _Operand(
            f"{_field(var, kind)} contains {needle.text}",
            lambda env, conv: (
                {env[var]["pos"]} if needle.value(env) in env[var]["content"] else set()
            ),
            lambda env, conv: {env[var]["pos"]},
        )
    if size > 0 and draw(st.integers(min_value=0, max_value=4)) == 0:
        return _Operand(
            'called_tools("x", "y")', lambda env, conv: set(), lambda env, conv: set(), ("x", "y")
        )
    name = draw(st.sampled_from(["x", "y"]))
    return _Operand(
        f'called_tool("{name}")',
        lambda env, conv: {e["pos"] for e in _elements(conv, "call") if e["name"] == name},
        lambda env, conv: {e["pos"] for e in _elements(conv, "call")},
    )


@st.composite
def conversations(draw) -> dict:
    """1 to 8 messages (the scanner rejects an empty list): user, assistant, system and
    developer turns (possibly empty text), assistant turns with one or two tool calls, and tool
    results answering earlier calls; plus zero to three tool definitions with distinct names."""
    messages = []
    unanswered: list[str] = []
    texts = st.sampled_from([*_WORDS, ""])
    for index in range(draw(st.integers(min_value=1, max_value=8))):
        kind = draw(st.sampled_from(["user", "assistant", "call", "tool", "system", "developer"]))
        if kind == "tool" and not unanswered:
            kind = "user"
        if kind == "call":
            calls = []
            for n in range(draw(st.integers(min_value=1, max_value=2))):
                call_id = f"c{index}_{n}"
                unanswered.append(call_id)
                calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": draw(st.sampled_from(["x", "y"])),
                            "arguments": json.dumps({"cmd": draw(st.sampled_from(_WORDS))}),
                        },
                    }
                )
            messages.append({"role": "assistant", "content": None, "tool_calls": calls})
        elif kind == "tool":
            call_id = unanswered.pop(draw(st.integers(min_value=0, max_value=len(unanswered) - 1)))
            messages.append({"role": "tool", "tool_call_id": call_id, "content": draw(texts)})
        else:
            messages.append({"role": kind, "content": draw(texts)})
    names = draw(st.lists(st.sampled_from(_TOOL_NAMES), max_size=3, unique=True))
    tools = [
        {"type": "function", "function": {"name": name, "description": draw(texts)}}
        for name in names
    ]
    return {"messages": messages, "tools": tools}


@st.composite
def captured(draw, rest: Callable[[bool], st.SearchStrategy[Rule]]) -> Rule:
    """``(any k in C where $p in k.field as v) and REST``: some value the pattern matches in
    ``C`` makes ``REST`` hold, where REST's element conditions may test ``... contains v``."""
    collection = draw(_collections())
    name = draw(st.sampled_from(sorted(_PATTERNS)))
    body = draw(rest(True))

    def values(conv) -> set[str]:
        return {v for e in collection.members(conv) for v in _pattern_matches(name, _text(e))}

    return Rule(
        f"((any k in {collection.text} where {name} in {_field('k', collection.kind)} as v) "
        f"and {body.text})",
        lambda env, conv: any(body.holds({**env, "v": v}, conv) for v in values(conv)),
        tuple(dict.fromkeys((_definition(name), *body.strings))),
    )


@st.composite
def flow_rules(draw, captured_value: bool = False) -> Rule:
    """A flow at the top level or as (part of) one comprehension body, drawn often enough to
    reach every flow operator and operand shape."""
    if not captured_value and draw(st.integers(min_value=0, max_value=3)) == 0:
        return draw(captured(lambda c: flow_rules(captured_value=c)))
    if draw(st.booleans()):
        return draw(_rules(size=1, choice="flow", captured=captured_value))
    collection = draw(_collections())
    kind = collection.kind
    flow = draw(_rules((("v0", kind),), depth=0, size=1, choice="flow", captured=captured_value))
    sibling = draw(
        st.one_of(st.none(), _rules((("v0", kind),), depth=0, size=0, captured=captured_value))
    )
    body = (
        flow
        if sibling is None
        else Rule(
            f"({sibling.text} and {flow.text})",
            lambda env, conv: sibling.holds(env, conv) and flow.holds(env, conv),
            _merged(sibling, flow),
        )
    )
    return Rule(
        f"(any v0 in {collection.text} where {body.text})",
        lambda env, conv: any(
            body.holds({**env, "v0": element}, conv) for element in collection.members(conv)
        ),
        body.strings,
    )


@st.composite
def two_join_rules(draw) -> Rule:
    """``any v0 in S where`` two comprehensions, each joining its element to ``v0`` with
    ``v0`` on either side, combined by ``and``, ``or``, ``and not`` or a flow, the second one
    ``any`` or (outside a flow) ``all``: two joins that may need two captured values at once,
    or a value read under ``not`` or ``all``, when both share the containing or the contained
    element ``v0``."""
    outer = draw(_collections())
    scope = (("v0", outer.kind),)

    def joined(var: str):
        coll = draw(_collections())
        join = draw(_rules((*scope, (var, coll.kind)), 0, 0, choice="join"))
        return coll, join

    (c1, j1), (c2, j2) = joined("v1"), joined("v2")
    shape = draw(st.sampled_from(["and", "or", "after", "and not"]))
    every = shape != "after" and draw(st.booleans())
    first = f"(any v1 in {c1.text} where {j1.text})"
    second = f"({'all' if every else 'any'} v2 in {c2.text} where {j2.text})"

    def hits(env, conv, coll, join, var) -> set[int]:
        return {e["pos"] for e in coll.members(conv) if join.holds({**env, var: e}, conv)}

    def body(env, conv) -> bool:
        a, b = hits(env, conv, c1, j1, "v1"), hits(env, conv, c2, j2, "v2")
        held = len(b) == len(c2.members(conv)) if every else bool(b)
        if shape == "and":
            return bool(a) and held
        if shape == "or":
            return bool(a) or held
        if shape == "and not":
            return bool(a) and not held
        return any(i < j for i in b for j in a)  # first after second

    text = f"({first} {shape} {second})"
    return Rule(
        f"(any v0 in {outer.text} where {text})",
        lambda env, conv: any(body({**env, "v0": e}, conv) for e in outer.members(conv)),
    )


# Plain rules, and plain rules over a captured value.
rules = st.one_of(_rules(), captured(lambda c: _rules(captured=c)))
