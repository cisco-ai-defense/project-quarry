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

"""Round-trip and safety regressions for public condition rendering."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

import quarry.engine.lower as lower_module
import quarry.plugin.visualization as visualization_module
import quarry.types.dsl.render as render_module
from quarry.dsl.parser import DSLParser
from quarry.engine.compiler import compile_rule
from quarry.plugin.service import QuarryPlugin
from quarry.plugin.visualization import build_rule_visualization
from quarry.types.dsl import (
    AllComprehension,
    AnyComprehension,
    Assignment,
    BinaryOp,
    ConditionalValue,
    ConditionNode,
    ContainsAnyOp,
    FieldAccess,
    FilterComprehension,
    FlowOp,
    FunctionCall,
    Identifier,
    Literal,
    MapComprehension,
    ModelCall,
    ParameterRef,
    PatternMatch,
    StringCount,
    StringPosition,
    StringSetQuantifier,
    UnaryOp,
)
from quarry.types.dsl.render import ConditionRenderError, render_condition

_PARSER = DSLParser()


def _parse(expression: str):
    return _PARSER.parse(f'rule "Synthetic"\ncondition: {expression}\n')


def _round_trip(expression: str) -> str:
    condition = _parse(expression).condition
    assert condition is not None
    rendered = render_condition(condition)
    assert _parse(rendered).condition == condition
    return rendered


@pytest.mark.parametrize(
    "expression",
    [
        "not (true and false)",
        '(any item in messages where item.role == "user") and true',
        'true and (all item in messages where item.content contains "ok")',
        '(not (any item in messages where item.role == "tool")) and true',
        "(any inner in messages where true) if false else true",
        '("token" in user_messages[0].content as found) and found in tool_responses[-1].content',
        'messages[0].content contains_any ["alpha", "雪", ["nested", 3]]',
        "messages[0].content contains $needles",
        'any item in messages where item matches {role == "user" or content contains "雪"}',
        "2 of ($first, $other*) in user_messages[0].content",
        "@first in user_messages[0].content and #other* > 1",
        'messages.called_tool("lookup") or conversation.extract(messages, 0, 1) == "x"',
        'model("judge").classify(messages[:1] + messages[-1:]) > 0.5',
        '[selected for item in messages if item.role == "user"]',
        'join([item.content for item in messages if item.role != "tool"], "\\n") == "x"',
        "any item in (any other in messages where true) where item == true",
        'user_messages[0].content ~ "\\d+\\s雪\\\\\\\\path"',
        "true right_after false",
        "true before(within: 3) false",
    ],
)
def test_supported_condition_shapes_round_trip(expression: str) -> None:
    _round_trip(expression)


@pytest.mark.parametrize(
    "expression",
    [
        'messages[0].content contains_any ["plain", ["nested"]]',
        '"plain" in ["plain", ["nested"]]',
    ],
)
def test_parser_normalizes_executable_array_primitives(expression: str) -> None:
    condition = _parse(expression).condition
    assert condition is not None
    values = condition.values if isinstance(condition, ContainsAnyOp) else condition.right.value

    assert type(values) is list
    assert type(values[0]) is str
    assert type(values[1]) is list
    assert type(values[1][0]) is str


@given(
    st.lists(st.booleans(), min_size=1, max_size=40),
    st.sampled_from(("and", "or")),
)
def test_flat_boolean_properties_round_trip(values: list[bool], operator: str) -> None:
    expression = f" {operator} ".join(str(value).lower() for value in values)
    _round_trip(expression)


@given(
    st.lists(
        st.sampled_from(("plain", "雪", "\\", '"', "\n", "\t", "🙂")),
        min_size=0,
        max_size=20,
    )
)
def test_literal_unicode_and_escaping_round_trip(parts: list[str]) -> None:
    value = "".join(parts)
    expression = f"messages[0].content == {json.dumps(value)}"
    _round_trip(expression)


@given(
    st.lists(
        st.sampled_from((r"\d", r"\s", r"\\", r"\.", r"\"", "雪", "[a-z]", "+")),
        min_size=1,
        max_size=12,
    )
)
def test_regex_spelling_round_trips_without_escape_growth(parts: list[str]) -> None:
    spelling = "".join(parts)
    rendered = _round_trip(f'messages[0].content ~ "{spelling}"')
    assert f'~ "{spelling}"' in rendered


@pytest.mark.parametrize("operator", ["and", "or"])
def test_long_flat_boolean_chain_does_not_render_as_nested_parentheses(
    operator: str,
) -> None:
    condition: ConditionNode = Literal(True)
    for index in range(1_000):
        condition = BinaryOp(operator, condition, Literal(index % 2 == 0))

    rendered = render_condition(condition)
    reparsed = _parse(rendered).condition

    assert rendered.count("(") == 1
    assert len(rendered) < 11_000
    terms = []
    while isinstance(reparsed, BinaryOp) and reparsed.operator == operator:
        terms.append(reparsed.right)
        reparsed = reparsed.left
    terms.append(reparsed)
    assert len(terms) == 1_001


def test_regex_and_pattern_block_have_parseable_public_spelling() -> None:
    regex = _round_trip(r'user_messages[0].content ~ "\d+\s雪"')
    patterns = _round_trip(
        'any item in messages where item matches {role == "user" or content contains "雪"}'
    )

    assert regex.endswith(r'~ "\d+\s雪")')
    assert "{role ==" in patterns
    assert "{(" not in patterns


def test_decision_and_plugin_labels_use_round_trippable_rendering() -> None:
    rule = _parse('(any item in messages where item.content ~ "\\d+雪") and called_tool("lookup")')
    compiled = compile_rule(rule)
    assert compiled.ok
    assert compiled.artifact is not None

    for decision in compiled.artifact.decision_tree.nodes:
        assert _parse(decision.label).condition is not None

    graph = build_rule_visualization(rule, compiled.artifact)
    root = next(node for node in graph.nodes if node.id == graph.root)
    assert _parse(root.label).condition == rule.condition


@dataclass
class _UnsupportedNode(ConditionNode):
    payload: str


@pytest.mark.parametrize(
    ("node", "kwargs", "message"),
    [
        (_UnsupportedNode("x"), {}, "unsupported condition node"),
        (Literal({"not": "a DSL literal"}), {}, "literal type dict"),
        (Literal("too much"), {"max_output": 4}, "output characters"),
        (UnaryOp("not", Literal(True)), {"max_nodes": 1}, "condition nodes"),
    ],
)
def test_rendering_fails_closed_for_unsupported_or_limited_inputs(
    node: ConditionNode, kwargs: dict[str, int], message: str
) -> None:
    with pytest.raises(ConditionRenderError, match=message):
        render_condition(node, **kwargs)


def test_rendering_rejects_depth_and_cycles() -> None:
    nested: ConditionNode = Literal(True)
    for _ in range(6):
        nested = UnaryOp("not", nested)
    with pytest.raises(ConditionRenderError, match="nested levels"):
        render_condition(nested, max_depth=3)

    cyclic = UnaryOp("not", Literal(True))
    cyclic.operand = cyclic
    with pytest.raises(ConditionRenderError, match="cycle"):
        render_condition(cyclic)


def test_unary_under_equality_keeps_the_authored_operator_tree() -> None:
    expression = '(not model("judge").classify(messages) > 0.5) == false'
    condition = _parse(expression).condition
    assert condition is not None

    rendered = render_condition(condition)

    assert rendered == '((not (model("judge").classify(messages) > 0.5)) == false)'
    assert _parse(rendered).condition == condition


_NodeFactory = Callable[[], ConditionNode]
_ParentFactory = Callable[[ConditionNode], ConditionNode]


def _map_node() -> MapComprehension:
    return MapComprehension(
        FieldAccess(Identifier("item"), ["content"]),
        "item",
        Identifier("messages"),
        Literal(True),
    )


_OPERATOR_CHILDREN: tuple[tuple[str, _NodeFactory], ...] = (
    ("literal", lambda: Literal("leaf")),
    ("parameter", lambda: ParameterRef("threshold")),
    ("not", lambda: UnaryOp("not", Literal(True))),
    ("conditional", lambda: ConditionalValue(Literal(1), Literal(True), Literal(2))),
    ("or", lambda: BinaryOp("or", Literal(True), Literal(False))),
    ("and", lambda: BinaryOp("and", Literal(True), Literal(False))),
    ("flow", lambda: FlowOp("after", Literal(True), Literal(False))),
    ("equality", lambda: BinaryOp("=", Literal(1), Literal(2))),
    ("contains", lambda: BinaryOp("contains", Literal("a"), Literal("b"))),
    ("regex", lambda: BinaryOp("~", Literal("a"), Literal("b"))),
    ("membership", lambda: BinaryOp("in", Literal("a"), Literal("b"))),
    ("addition", lambda: BinaryOp("+", Literal(1), Literal(2))),
    ("subtraction", lambda: BinaryOp("-", Literal(1), Literal(2))),
    ("multiplication", lambda: BinaryOp("*", Literal(1), Literal(2))),
    ("alias", lambda: Assignment(Literal(1), "bound")),
    ("contains-any", lambda: ContainsAnyOp(Literal("text"), ["x"])),
    (
        "any-comprehension",
        lambda: AnyComprehension("item", Identifier("messages"), Literal(True)),
    ),
    (
        "all-comprehension",
        lambda: AllComprehension("item", Identifier("messages"), Literal(True)),
    ),
    (
        "filter-comprehension",
        lambda: FilterComprehension(
            "item", Identifier("messages"), Literal(True), output_variable="item"
        ),
    ),
    ("map-comprehension", _map_node),
    ("field", lambda: FieldAccess(Identifier("item"), ["content"])),
    ("index", lambda: FunctionCall("_index", [Identifier("messages"), Literal(0)])),
    (
        "model-call",
        lambda: ModelCall("judge", "classify", [Identifier("messages")], "float"),
    ),
    (
        "pattern-match",
        lambda: BinaryOp(
            "matches",
            Identifier("item"),
            PatternMatch([BinaryOp("=", Identifier("role"), Literal("user"))]),
        ),
    ),
    ("string-position", lambda: StringPosition("first")),
    ("string-count", lambda: StringCount("first")),
    ("string-set", lambda: StringSetQuantifier("any", ["$first"])),
)


def _binary_parent(operator: str, side: str) -> _ParentFactory:
    other = Literal("other")
    if side == "left":
        return lambda child: BinaryOp(operator, child, other)
    return lambda child: BinaryOp(operator, other, child)


def _flow_parent(operator: str, side: str) -> _ParentFactory:
    if side == "left":
        return lambda child: FlowOp(operator, child, Literal(True))
    return lambda child: FlowOp(operator, Literal(True), child)


_PARENT_POSITIONS: tuple[tuple[str, _ParentFactory], ...] = (
    ("not.operand", lambda child: UnaryOp("not", child)),
    (
        "conditional.chosen",
        lambda child: ConditionalValue(child, Literal(True), Literal(0)),
    ),
    (
        "conditional.predicate",
        lambda child: ConditionalValue(Literal(1), child, Literal(0)),
    ),
    (
        "conditional.fallback",
        lambda child: ConditionalValue(Literal(1), Literal(True), child),
    ),
    ("alias.expression", lambda child: Assignment(child, "outer")),
    (
        "any.collection",
        lambda child: AnyComprehension("x", child, Literal(True)),
    ),
    (
        "any.condition",
        lambda child: AnyComprehension("x", Identifier("messages"), child),
    ),
    (
        "filter.collection",
        lambda child: FilterComprehension("x", child, Literal(True), output_variable="x"),
    ),
    (
        "filter.condition",
        lambda child: FilterComprehension("x", Identifier("messages"), child, output_variable="x"),
    ),
    (
        "map.expression",
        lambda child: MapComprehension(child, "x", Identifier("messages"), Literal(True)),
    ),
    (
        "map.collection",
        lambda child: MapComprehension(
            FieldAccess(Identifier("x"), ["content"]),
            "x",
            child,
            Literal(True),
        ),
    ),
    (
        "map.condition",
        lambda child: MapComprehension(
            FieldAccess(Identifier("x"), ["content"]),
            "x",
            Identifier("messages"),
            child,
        ),
    ),
    ("field.base", lambda child: FieldAccess(child, ["value"])),
    ("index.base", lambda child: FunctionCall("_index", [child, Literal(0)])),
    (
        "index.argument",
        lambda child: FunctionCall("_index", [Identifier("messages"), child]),
    ),
    ("call.argument", lambda child: FunctionCall("len", [child])),
    (
        "model.argument",
        lambda child: ModelCall("judge", "classify", [child], "float"),
    ),
    ("contains-any.left", lambda child: ContainsAnyOp(child, ["x"])),
    *tuple(
        (f"{operator}.{side}", _binary_parent(operator, side))
        for operator in (
            "or",
            "and",
            "=",
            "!=",
            ">",
            "<",
            ">=",
            "<=",
            "contains",
            "~",
            "in",
            "+",
            "-",
            "*",
        )
        for side in ("left", "right")
    ),
    *tuple(
        (f"{operator}.{side}", _flow_parent(operator, side))
        for operator in ("after", "until")
        for side in ("left", "right")
    ),
)


_OPERATOR_CASES = tuple(
    (child_name, child_factory, parent_name, parent_factory)
    for child_name, child_factory in _OPERATOR_CHILDREN
    for parent_name, parent_factory in _PARENT_POSITIONS
    if not (parent_name == "field.base" and child_name in {"parameter", "field"})
)


@pytest.mark.parametrize(
    ("child_name", "child_factory", "parent_name", "parent_factory"),
    _OPERATOR_CASES,
    ids=[f"{parent_name}-{child_name}" for child_name, _, parent_name, _ in _OPERATOR_CASES],
)
def test_every_supported_parent_child_operator_shape_round_trips(
    child_name: str,
    child_factory: _NodeFactory,
    parent_name: str,
    parent_factory: _ParentFactory,
) -> None:
    del child_name, parent_name
    expected = parent_factory(child_factory())

    rendered = render_condition(expected)

    assert _parse(rendered).condition == expected


def _cyclic_unary_binary() -> ConditionNode:
    unary = UnaryOp("not", Literal(True))
    binary = BinaryOp("and", unary, Literal(True))
    unary.operand = binary
    return binary


def _cyclic_binary() -> ConditionNode:
    node = BinaryOp("and", Literal(True), Literal(False))
    node.left = node
    return node


def _cyclic_comprehension() -> ConditionNode:
    node = AnyComprehension("item", Identifier("messages"), Literal(True))
    node.condition = UnaryOp("not", node)
    return node


def _cyclic_alias() -> ConditionNode:
    node = Assignment(Literal(True), "value")
    node.expression = BinaryOp("+", Literal(1), node)
    return node


@pytest.mark.parametrize(
    "factory",
    [_cyclic_binary, _cyclic_unary_binary, _cyclic_comprehension, _cyclic_alias],
    ids=["binary", "unary-binary", "comprehension", "alias"],
)
def test_render_preflight_rejects_indirect_cycles(factory: _NodeFactory) -> None:
    with pytest.raises(ConditionRenderError, match="cycle"):
        render_condition(factory())


def test_render_preflight_accepts_a_shared_acyclic_dag() -> None:
    shared = UnaryOp("not", Literal(False))
    root = BinaryOp("and", shared, shared)

    rendered = render_condition(root)

    assert _parse(rendered).condition == root


@pytest.mark.parametrize(
    "value",
    ['quote"', "back\\slash", "雪🙂", "\x00\b\f\n\r\t\x1f"],
)
def test_incremental_string_escaping_preserves_dsl_values_in_ascii_spelling(value: str) -> None:
    rendered = render_condition(Literal(value))
    expected = '"\\u96ea\\U0001f642"' if value == "雪🙂" else json.dumps(value)
    assert rendered == expected
    assert _parse(rendered).condition == Literal(value)


def test_huge_string_is_rejected_before_the_encoder_runs(monkeypatch) -> None:
    calls = 0

    def counted_encoder(value: str):
        nonlocal calls
        calls += 1
        yield from _json_chunks(value)

    _json_chunks = render_module._json_escape_chunks
    monkeypatch.setattr(render_module, "_json_escape_chunks", counted_encoder)

    with pytest.raises(ConditionRenderError, match="output characters"):
        render_condition(Literal("x" * 1_000_000), max_output=8)

    assert calls == 0


def test_incremental_encoder_keeps_each_temporary_chunk_bounded(monkeypatch) -> None:
    chunk_sizes: list[int] = []
    encoder = render_module._json_escape_chunks

    def counted_encoder(value: str):
        for chunk in encoder(value):
            chunk_sizes.append(len(chunk))
            yield chunk

    monkeypatch.setattr(render_module, "_json_escape_chunks", counted_encoder)
    value = ('"\\雪\n🙂' * 4_000) + "\x00"

    rendered = render_condition(Literal(value), max_output=500_000)

    assert rendered.isascii()
    assert _parse(rendered).condition == Literal(value)
    assert len(rendered) == 2 + sum(chunk_sizes)
    assert chunk_sizes
    assert max(chunk_sizes) <= render_module._ESCAPE_CHUNK_SIZE + 9


class _CountedText(str):
    iterations = 0

    def __iter__(self):
        for character in super().__iter__():
            type(self).iterations += 1
            yield character


def test_regex_subclass_is_rejected_before_content_validation() -> None:
    raw = _CountedText("x" * 1_000_000)
    _CountedText.iterations = 0
    condition = BinaryOp("~", Literal("text"), Literal(raw))

    with pytest.raises(ConditionRenderError, match="regex operand must be text"):
        render_condition(condition, max_output=16)

    assert _CountedText.iterations == 0


class _CountedPattern:
    calls = 0

    def fullmatch(self, value: str):
        type(self).calls += 1
        return True


@pytest.mark.parametrize(
    ("node", "pattern_name"),
    [
        (Identifier("x" * 1_000_000), "_IDENTIFIER"),
        (StringSetQuantifier("any", ["$" + "x" * 1_000_000]), "_STRING_VARIABLE"),
    ],
)
def test_huge_names_are_rejected_before_pattern_validation(
    monkeypatch, node: ConditionNode, pattern_name: str
) -> None:
    pattern = _CountedPattern()
    _CountedPattern.calls = 0
    monkeypatch.setattr(render_module, pattern_name, pattern)

    with pytest.raises(ConditionRenderError, match="output characters"):
        render_condition(node, max_output=16)

    assert _CountedPattern.calls == 0


@pytest.mark.parametrize(
    ("value", "max_output", "expected"),
    [
        (0, 1, "0"),
        (9_999, 4, "9999"),
        (-999, 4, "-999"),
        (10**63 - 1, 63, "9" * 63),
        (-(10**63 - 1), 64, "-" + "9" * 63),
    ],
)
def test_integer_preflight_renders_exact_boundary_fits(
    value: int, max_output: int, expected: str
) -> None:
    assert render_condition(Literal(value), max_output=max_output) == expected


@pytest.mark.parametrize("digits", [4_000, 100_000])
@pytest.mark.parametrize("sign", [1, -1])
def test_huge_integer_is_rejected_before_decimal_conversion(
    monkeypatch, digits: int, sign: int
) -> None:
    calls = 0

    def counted_conversion(value: int) -> str:
        nonlocal calls
        calls += 1
        return str(value)

    monkeypatch.setattr(render_module, "_integer_text", counted_conversion)
    value = sign * 10**digits

    with pytest.raises(ConditionRenderError, match="output characters"):
        render_condition(Literal(value), max_output=8)

    assert calls == 0


def test_integer_rendering_respects_the_runtime_conversion_limit() -> None:
    conversion_limit = sys.get_int_max_str_digits()
    if conversion_limit == 0:
        pytest.skip("runtime integer conversion limit is disabled")
    value = 10**conversion_limit

    with pytest.raises(ConditionRenderError, match="too large to render"):
        render_condition(Literal(value), max_output=conversion_limit + 2)

    assert sys.get_int_max_str_digits() == conversion_limit


class _CustomInteger(int):
    pass


@pytest.mark.parametrize("limit_name", ["max_output", "max_nodes", "max_depth"])
@pytest.mark.parametrize(
    "invalid",
    [
        True,
        False,
        0,
        -1,
        1.0,
        float("nan"),
        float("inf"),
        Decimal(1),
        Fraction(1, 1),
        _CustomInteger(1),
    ],
)
def test_public_render_limits_require_positive_plain_integers(
    limit_name: str, invalid: object
) -> None:
    kwargs: dict[str, Any] = {limit_name: invalid}
    with pytest.raises(ValueError, match="positive non-Boolean integers"):
        render_condition(Literal(True), **kwargs)


class _HostileText(str):
    calls = 0

    @classmethod
    def _trip(cls):
        cls.calls += 1
        raise AssertionError("hostile text method executed")

    def __len__(self):
        return self._trip()

    def __str__(self):
        return self._trip()

    def __iter__(self):
        return self._trip()

    def __hash__(self):
        return self._trip()

    def __eq__(self, _other):
        return self._trip()

    def __getitem__(self, _key):
        return self._trip()

    def startswith(self, *_args, **_kwargs):
        return self._trip()


class _HostileInteger(int):
    calls = 0

    @classmethod
    def _trip(cls):
        cls.calls += 1
        raise AssertionError("hostile integer method executed")

    def bit_length(self):
        return self._trip()

    def __str__(self):
        return self._trip()

    def __bool__(self):
        return self._trip()

    def __lt__(self, _other):
        return self._trip()

    def __ge__(self, _other):
        return self._trip()


class _HostileFloat(float):
    calls = 0

    @classmethod
    def _trip(cls):
        cls.calls += 1
        raise AssertionError("hostile float method executed")

    def __str__(self):
        return self._trip()

    def __float__(self):
        return self._trip()


class _HostileList(list):
    calls = 0

    @classmethod
    def _trip(cls):
        cls.calls += 1
        raise AssertionError("hostile list method executed")

    def __len__(self):
        return self._trip()

    def __iter__(self):
        return self._trip()

    def __getitem__(self, _key):
        return self._trip()

    def __bool__(self):
        return self._trip()


@pytest.mark.parametrize(
    "factory",
    [
        lambda value: Literal(value),
        lambda value: Literal("value", raw_value=value),
        lambda value: Literal([value]),
        lambda value: Identifier(value),
        lambda value: ParameterRef(value),
        lambda value: UnaryOp(value, Literal(True)),
        lambda value: BinaryOp(value, Literal(True), Literal(False)),
        lambda value: FlowOp(value, Literal(True), Literal(False)),
        lambda value: FieldAccess(Identifier("item"), [value]),
        lambda value: FunctionCall(value, []),
        lambda value: ModelCall(value, "classify", [Identifier("messages")], "float"),
        lambda value: ModelCall("judge", value, [Identifier("messages")], "float"),
        lambda value: ModelCall("judge", "classify", [Identifier("messages")], value),
        lambda value: AnyComprehension(value, Identifier("messages"), Literal(True)),
        lambda value: FilterComprehension(
            "item", Identifier("messages"), Literal(True), output_variable=value
        ),
        lambda value: Assignment(Literal(True), value),
        lambda value: StringPosition(value),
        lambda value: StringSetQuantifier(value, ["$first"]),
        lambda value: StringSetQuantifier("any", [value]),
        lambda value: ContainsAnyOp(Literal("text"), [value]),
    ],
)
def test_text_subclasses_are_rejected_without_invoking_overrides(
    factory: Callable[[object], ConditionNode],
) -> None:
    value = _HostileText("hostile")
    _HostileText.calls = 0

    with pytest.raises(ConditionRenderError):
        render_condition(factory(value))

    assert _HostileText.calls == 0


@pytest.mark.parametrize(
    "node",
    [
        Literal(_HostileInteger(7)),
        FlowOp("after", Literal(True), Literal(False), max_distance=_HostileInteger(2)),
        StringSetQuantifier(_HostileInteger(1), ["$first"]),
    ],
)
def test_integer_subclasses_are_rejected_without_invoking_overrides(
    node: ConditionNode,
) -> None:
    _HostileInteger.calls = 0

    with pytest.raises(ConditionRenderError):
        render_condition(node)

    assert _HostileInteger.calls == 0


def test_float_subclass_is_rejected_without_invoking_overrides() -> None:
    _HostileFloat.calls = 0

    with pytest.raises(ConditionRenderError):
        render_condition(Literal(_HostileFloat(0.5)))

    assert _HostileFloat.calls == 0


@pytest.mark.parametrize(
    "factory",
    [
        lambda value: Literal(value),
        lambda value: FunctionCall("len", value),
        lambda value: FieldAccess(Identifier("item"), value),
        lambda value: PatternMatch(value),
        lambda value: StringSetQuantifier("any", value),
        lambda value: ContainsAnyOp(Literal("text"), value),
    ],
)
def test_list_subclasses_are_rejected_without_invoking_overrides(
    factory: Callable[[object], ConditionNode],
) -> None:
    value = _HostileList([Literal(True)])
    _HostileList.calls = 0

    with pytest.raises(ConditionRenderError):
        render_condition(factory(value))

    assert _HostileList.calls == 0


def test_visualization_validates_before_iterating_hostile_children() -> None:
    arguments = _HostileList([Literal(True)])
    rule = _parse("true")
    rule.condition = FunctionCall("len", arguments)
    _HostileList.calls = 0

    with pytest.raises(ValueError, match="^rule condition cannot be rendered safely$"):
        build_rule_visualization(rule)

    assert _HostileList.calls == 0


def test_primitive_subclasses_are_rejected_as_public_limits_without_callbacks() -> None:
    value = _HostileInteger(8)
    _HostileInteger.calls = 0

    with pytest.raises(ValueError, match="positive non-Boolean integers"):
        render_condition(Literal(True), max_output=value)

    assert _HostileInteger.calls == 0


@pytest.mark.parametrize(
    "node",
    [
        FlowOp("after", Literal(True), Literal(False), max_distance=True),
        StringSetQuantifier(True, ["$first"]),
    ],
)
def test_boolean_does_not_pass_integer_ast_fields(node: ConditionNode) -> None:
    with pytest.raises(ConditionRenderError):
        render_condition(node)


def _deep_render_shape(kind: str, depth: int) -> ConditionNode:
    node: ConditionNode = Literal(True)
    for index in range(depth):
        if kind == "unary":
            node = UnaryOp("not", node)
        elif kind == "binary":
            node = BinaryOp("and", Literal(True), node)
        elif kind == "comprehension":
            node = AnyComprehension(f"item{index}", Identifier("messages"), node)
        elif kind == "alias":
            node = Assignment(node, f"value{index}")
        else:
            node = ConditionalValue(node, Literal(True), Literal(False))
    return node


def test_129_nested_nots_obey_only_the_configured_depth_bound() -> None:
    condition = _deep_render_shape("unary", 129)

    with pytest.raises(ConditionRenderError, match="nested levels"):
        render_condition(condition)

    rendered = render_condition(condition, max_depth=129)

    assert rendered.count("not ") == 129


@pytest.mark.parametrize(
    "kind",
    ["unary", "binary", "comprehension", "alias", "conditional"],
)
def test_supported_shape_depth_boundaries(kind: str) -> None:
    condition = _deep_render_shape(kind, 16)

    assert render_condition(condition, max_depth=16)
    with pytest.raises(ConditionRenderError, match="nested levels"):
        render_condition(condition, max_depth=15)


@pytest.mark.parametrize(
    "kind",
    ["unary", "binary", "comprehension", "alias", "conditional"],
)
def test_deep_supported_shapes_use_an_iterative_output_machine(kind: str) -> None:
    depth = 1_200
    condition = _deep_render_shape(kind, depth)

    rendered = render_condition(
        condition,
        max_depth=depth * 2 + 1,
        max_nodes=depth * 5 + 10,
    )

    assert rendered


def test_compile_rule_converts_renderer_failure_to_a_bounded_diagnostic(
    monkeypatch,
) -> None:
    def fail_render(_node):
        raise ConditionRenderError("unbounded detail " * 10_000)

    monkeypatch.setattr(lower_module, "render_condition", fail_render)

    result = compile_rule(_parse("true"))

    assert not result.ok
    assert [error.message for error in result.diagnostics.errors] == [
        "the condition cannot be rendered safely for compiler diagnostics"
    ]


def test_visualization_converts_renderer_failure_to_a_bounded_value_error(
    monkeypatch,
) -> None:
    def fail_render(_node):
        raise ConditionRenderError("unbounded detail " * 10_000)

    monkeypatch.setattr(visualization_module, "render_condition", fail_render)

    with pytest.raises(ValueError, match="^rule condition cannot be rendered safely$"):
        build_rule_visualization(_parse("true"))


def test_plugin_converts_renderer_failure_to_its_compile_diagnostic(
    monkeypatch,
) -> None:
    def fail_render(_node):
        raise ConditionRenderError("unbounded detail " * 10_000)

    monkeypatch.setattr(visualization_module, "render_condition", fail_render)

    report = QuarryPlugin().compile(
        rule_text='rule "Synthetic"\ncondition: true\n',
        visualize=True,
    )

    assert report["valid"] is False
    assert report["diagnostics"][0]["message"] == ("rule condition cannot be rendered safely")
