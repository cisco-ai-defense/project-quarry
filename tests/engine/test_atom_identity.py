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

"""Distinct model literals must remain distinct when atoms are shared."""

import pytest

from quarry.engine.atoms import AtomDomain, AtomInterner, Cost, ModelExpression, make_atom


def _model_literal_atom(value: bool | int | float):
    return make_atom(
        domain=AtomDomain.STATIC,
        selector="messages",
        value_expr_key="model_expression",
        predicate="external",
        cost=Cost.LLM,
        detector="model_expression",
        model_expression=ModelExpression("literal", value),
        evidence_policy="model",
    )


@pytest.mark.parametrize(
    ("first", "second"),
    [
        pytest.param(True, 1, id="true-versus-one"),
        pytest.param(False, 0, id="false-versus-zero"),
        pytest.param(1, 1.0, id="integer-versus-float"),
        pytest.param(0.0, -0.0, id="positive-versus-negative-zero"),
    ],
)
def test_distinct_model_literals_do_not_share_an_atom(first, second) -> None:
    first_spec = _model_literal_atom(first)
    second_spec = _model_literal_atom(second)
    assert first_spec.atom_id != second_spec.atom_id

    interner = AtomInterner()
    assert interner.intern(first_spec) is first_spec
    assert interner.intern(second_spec) is second_spec
    assert len(interner) == 2
    assert interner.intern(_model_literal_atom(first)) is first_spec
