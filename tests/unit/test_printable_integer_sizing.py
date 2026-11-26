# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""JSON byte budgets admit printable integers without an arbitrary digit cutoff."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

import quarry
from quarry import InputAdapter
from quarry.config import QuarryConfig
from quarry.safety.limits import SafetyLimits, json_size
from quarry.scanner import DSLScanner


@pytest.mark.parametrize("digits", [1, 3999, 4000, 4001, 4200, 4300])
@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize("margin", [-1, 0, 1])
def test_printable_integer_uses_exact_serialized_size(digits, negative, margin):
    integer = 10 ** (digits - 1) * (-1 if negative else 1)
    value = {"number": [integer, integer]}
    expected = len(json.dumps(value, separators=(",", ":")).encode())
    budget = expected + margin
    assert json_size(value, budget) == min(expected, budget + 1)


@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize("margin", [-1, 0, 1])
def test_public_native_input_admits_a_printable_integer(negative, margin):
    integer = 10**4199 * (-1 if negative else 1)
    native = {"number": integer}
    size = len(json.dumps(native, separators=(",", ":")).encode())
    calls = []

    class Native(InputAdapter):
        name = "native"
        version = "1"
        sections = ("numbers.state",)

        def convert(self, value, wanted):
            calls.append(value["number"])
            assert wanted == frozenset({"numbers.state"})
            return {"numbers.state": {"enabled": True}}

    scanner = DSLScanner(
        QuarryConfig(
            quarry_rule_texts=('rule "Read" condition: numbers.state.enabled == true',),
            inputs={
                "numbers.state": {
                    "single": True,
                    "facts": {"enabled": {"type": "boolean"}},
                }
            },
            adapters={"raw": {"adapter": "native"}},
            safety_limits=SafetyLimits(rule_timeout=None, max_total_size=size + margin),
        ),
        input_adapters={"native": Native()},
    )
    result = scanner.scan_data({"raw": native})
    if margin < 0:
        assert result.completion_state == "invalid"
        assert "max_total_size" in result.validation_error
        assert not calls
    else:
        assert result.completion_state == "complete"
        assert result.rule_matches == ["Read"]
        assert calls == [integer]


@pytest.mark.parametrize("value", [True, False, [True, False], {"flags": [False, True] * 8}])
@pytest.mark.parametrize("margin", [-1, 0, 1])
def test_booleans_keep_their_json_spelling_when_sizing(value, margin):
    expected = len(json.dumps(value, separators=(",", ":")))
    budget = expected + margin
    assert json_size(value, budget) == min(expected, budget + 1)


@pytest.mark.parametrize("margin", [-1, 0, 1])
def test_native_boolean_payload_is_sized_before_adapter_dispatch(margin):
    native = {"flags": [False, True] * 512}
    size = len(json.dumps(native, separators=(",", ":")))
    calls = []

    class Native(InputAdapter):
        name = "native"
        version = "1"
        sections = ("numbers.state",)

        def convert(self, value, wanted):
            calls.append(value)
            return {"numbers.state": {"enabled": True}}

    scanner = DSLScanner(
        QuarryConfig(
            quarry_rule_texts=('rule "Read" condition: numbers.state.enabled == true',),
            inputs={"numbers.state": {"single": True, "facts": {"enabled": {"type": "boolean"}}}},
            adapters={"raw": {"adapter": "native"}},
            safety_limits=SafetyLimits(rule_timeout=None, max_total_size=size + margin),
        ),
        input_adapters={"native": Native()},
    )
    result = scanner.scan_data({"raw": native})
    if margin < 0:
        assert result.completion_state == "invalid"
        assert "max_total_size" in result.validation_error
        assert not calls
    else:
        assert result.completion_state == "complete"
        assert result.rule_matches == ["Read"]
        assert calls == [native]


@pytest.mark.parametrize("negative", [False, True])
def test_integer_subclass_is_measured_without_host_overrides(negative):
    class HostInteger(int):
        def __str__(self):
            raise AssertionError("host string conversion must not run")

        __repr__ = __str__
        __int__ = __str__
        __abs__ = __str__
        bit_length = __str__

    number = 10**4199 * (-1 if negative else 1)
    expected = len(str(number))
    assert json_size(HostInteger(number), expected) == expected
    assert json_size(HostInteger(number), expected - 1) == expected


@pytest.mark.parametrize(
    ("digit_limit", "digits"), [(640, 640), (640, 641), (10000, 7000), (0, 7000)]
)
def test_active_interpreter_conversion_limit_is_respected_in_isolation(digit_limit, digits):
    script = """
import json,sys
sys.path.insert(0, sys.argv[1])
from quarry.safety.limits import json_size
sys.set_int_max_str_digits(int(sys.argv[2]))
digits=int(sys.argv[3]); number=10**(digits-1)
try:
    expected=len(json.dumps(number))
except ValueError:
    expected=digits+11
assert json_size(number,digits+10)==expected
assert json_size(-number,digits+11)==(digits+12 if expected==digits+11 else digits+1)
assert json_size(10**100000,16)==17
print(json.dumps({'digit_limit':sys.get_int_max_str_digits(),'digits':digits,'passed':True}))
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            script,
            str(Path(quarry.__file__).resolve().parent.parent),
            str(digit_limit),
            str(digits),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "digit_limit": digit_limit,
        "digits": digits,
        "passed": True,
    }
