# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""A closed expression must not hide registered host reads or their failures."""

from typing import ClassVar

import pytest

from quarry import ToolCall
from quarry.config import QuarryConfig
from quarry.safety.limits import SafetyLimits
from quarry.scanner import DSLScanner
from quarry.types.exceptions import InvalidInputError


@pytest.mark.parametrize("explain", [False, True, "support"])
@pytest.mark.parametrize("fail_later", [False, True])
def test_registered_call_guards_keep_every_required_invocation(explain, fail_later):
    class GuardedCall(ToolCall):
        reads: ClassVar[list[str]] = []

        def guard_read(self, field):
            super().guard_read(field)
            if field == "name":
                self.reads.append(self.id)
                if fail_later and len(self.reads) == 2:
                    raise InvalidInputError("synthetic host read failure")

    source = (
        'rule "Read" condition: count([c for c in assistant_messages[-1].tool_calls '
        'if assistant_messages[-1].tool_calls[0].name == "lookup"]) == 8'
    )
    data = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": f"c{index}",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                    for index in range(8)
                ],
            }
        ]
    }
    scanner = DSLScanner(
        QuarryConfig(
            config_version=2,
            quarry_rule_texts=(source,),
            safety_limits=SafetyLimits(rule_timeout=None),
        ),
        tool_call_types={("function", "lookup"): GuardedCall},
    )
    result = scanner.scan_data(data, enable_explain=explain)
    assert GuardedCall.reads and set(GuardedCall.reads) == {"c0"}
    if fail_later:
        assert not result.is_complete and result.rule_matches == []
        assert len(GuardedCall.reads) >= 2
    else:
        assert result.is_complete and result.rule_matches == ["Read"]
        assert len(GuardedCall.reads) >= 8


@pytest.mark.asyncio
async def test_registered_host_values_do_not_prevent_independent_literal_work_reuse(monkeypatch):
    from quarry.engine.atoms import ModelExpression as Expr
    from quarry.integrations import model_expression
    from quarry.types.context import ScanContext

    class Registered(ToolCall):
        pass

    ctx = ScanContext(
        [{"role": "user", "content": "synthetic"}],
        tool_call_types={("function", "lookup"): Registered},
    )
    expression = Expr("+", children=(Expr("literal", 2), Expr("literal", 3)))
    reads = []
    original = model_expression._evaluate_uncached_model_expression

    async def observed(node, *args, **kwargs):
        if node is expression:
            reads.append(True)
        return await original(node, *args, **kwargs)

    monkeypatch.setattr(model_expression, "_evaluate_uncached_model_expression", observed)
    for _ in range(2):
        assert await model_expression._evaluate_filter(expression, ctx) == 5
    assert reads == [True]
    ctx.close_message_values()
