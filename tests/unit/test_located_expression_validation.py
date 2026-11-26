# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Prepared located-text expressions retain the current closed graph contract."""

import pytest

from quarry.engine.atoms import ModelExpression as E
from quarry.engine.atoms import validate_model_expression

OWNER = E("binding", "entry")
ROLE = E("literal", "user")
ORDER = E("literal", "title\x1fbody")
BODY = E("field", "body", (OWNER,))


@pytest.mark.parametrize(
    "expression",
    [
        E("input_message", "memory.notes", (ROLE, ORDER, BODY)),
        E("input_state", "skill.instructions"),
        E(
            "span_count",
            None,
            (
                E(
                    "string_spans",
                    "$p",
                    (E("messages"), E("string_patterns", "0000", (E("literal", "needle"),))),
                ),
            ),
        ),
    ],
)
def test_located_expression_shapes_are_admitted(expression):
    validate_model_expression(expression)


@pytest.mark.parametrize(
    "expression",
    [
        E("input_message", "memory", (ROLE, ORDER, BODY)),
        E("input_message", "memory.notes", (E("literal", "tool"), ORDER, BODY)),
        E("input_message", "memory.notes", (ROLE, E("literal", "body\x1fbody"), BODY)),
        E("input_message", "memory.notes", (ROLE, ORDER, E("field", "secret", (OWNER,)))),
        E(
            "input_message",
            "memory.notes",
            (ROLE, ORDER, BODY, E("field", "title", (E("binding", "other"),))),
        ),
        E("string_spans", "$p", (E("messages"), E("literal", "not-declared-patterns"))),
        E("span_count", None, ()),
        E("span_any", "not an identifier", (E("messages"), E("literal", True))),
        E("extract_windows", None, (E("messages"),)),
        E("span_text", "arguments..secret", (OWNER,)),
        E("input_state", "not a section"),
    ],
)
def test_located_expression_shapes_refuse_malformed_graphs(expression):
    with pytest.raises((ValueError, TypeError)):
        validate_model_expression(expression)
