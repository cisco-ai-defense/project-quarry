# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Prepared local-call graphs keep the compiler's source and predicate contracts."""

import pytest

from quarry.engine.atoms import ModelExpression as E
from quarry.engine.atoms import validate_model_expression
from quarry.policy_serialization import _required_model_expression

CALLS = E("field", "tool_calls", (E("binding", "message"),))


def wire(expression):
    return {
        "kind": expression.kind,
        "value": expression.value,
        "children": [wire(child) for child in expression.children],
    }


@pytest.mark.parametrize("serialized", [False, True])
@pytest.mark.parametrize(
    "expression",
    [
        E("call_count", False, (CALLS,)),
        E("call_count", None, (E("tools"),)),
        E("call_filter", "call.id", (CALLS, E("literal", True))),
        E("call_filter", "call", (CALLS, E("literal", 1))),
        E("call_any", "call", (CALLS, E("literal", "private value"))),
        E("call_all", "call", (CALLS, E("call_count", None, (CALLS,)))),
        E("call_count", None, (E("field", "tool_calls", (E("literal", "private holder"),)),)),
        E("call_count", None, (E("slice", None, (CALLS, E("literal", True), E("literal", None))),)),
    ],
)
def test_invalid_local_call_graphs_fail_before_evaluation(serialized, expression):
    with pytest.raises((ValueError, TypeError)):
        if serialized:
            _required_model_expression(wire(expression), "synthetic.local_calls")
        else:
            validate_model_expression(expression)


@pytest.mark.parametrize("kind", ["call_count", "call_filter", "call_any", "call_all"])
def test_valid_native_source_graphs_survive_exact_wire_loading(kind):
    expression = E(
        kind,
        None if kind == "call_count" else "call",
        (CALLS,) if kind == "call_count" else (CALLS, E("literal", True)),
    )
    validate_model_expression(expression)
    assert _required_model_expression(wire(expression), "synthetic.local_calls") == expression
