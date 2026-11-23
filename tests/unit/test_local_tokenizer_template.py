# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Safe rendering matches supported Jinja syntax and rejects partial request counts."""

import json
import tracemalloc
from dataclasses import replace
from time import monotonic

import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment

from quarry.integrations.local_tokenizer_input import snapshot_request
from quarry.integrations.local_tokenizer_template import render_local_request
from quarry.integrations.local_tokenizer_worker import _Abandoned, _Expired, _Job
from quarry.types.exceptions import (
    LocalTokenizerInvalidInputError,
    LocalTokenizerLimitError,
    LocalTokenizerUnavailableError,
)
from quarry.types.local_tokenizer import LocalTokenizerLimits
from tests.unit.test_local_tokenizer_types import config

MESSAGES = [
    {"role": "user", "content": "  synthetic é👩\u200d💻  "},
    {"role": "assistant", "content": "reply"},
]
TOOLS = [
    {
        "type": "function",
        "function": {"name": "synthetic", "description": "<é>&'", "parameters": {"type": "object"}},
    }
]


def render(template, *, messages=None, tools=None, options=None, job=None):
    cfg = config(template=template, **(options or {}))
    job = job or _Job(monotonic() + 5)
    messages, tools = snapshot_request(
        MESSAGES if messages is None else messages, [] if tools is None else tools, cfg, job
    )
    return render_local_request(cfg, messages, tools, job)


@pytest.mark.parametrize(
    "template",
    [
        "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}",
        "{% for m in messages if m.role == 'user' %}{{ loop.index }}{{ m.content|trim|upper }}{% else %}empty{% endfor %}",
        "{% set prefix = 'value=' %}{{ prefix + messages[0]['content'] }}",
        "{{ messages[-1].get('content')|lower }}{{ messages[0].get('missing', 'fallback') }}",
        "{% for key, value in messages[0].items() %}{{ key }}={{ value }}{% endfor %}",
        "{% if add_generation_prompt %}assistant{% elif messages|length > 1 %}many{% else %}one{% endif %}",
        "{{ messages[0].content|replace('synthetic', 'public') }}",
        "{% for i in range(2) %}{{ messages[i].content }}{% endfor %}",
        "{{ 'yes' if messages else 'no' }}{{ unknown|default('known') }}",
        "{{ messages[0].content[2:-2] }}{{ ['a', 'b']|join('::') }}",
        "{{ (1 + 2) == 3 and not false }}{{ 'user' in ['user', 'assistant'] }}",
        "{{ 'a' ~ 17 }}{{ -1 < 0 <= 1 }}",
        "{% for m in [] %}bad{% else %}empty{% endfor %}",
    ],
)
def test_supported_syntax_matches_fresh_native_jinja_rendering(template):
    expected = (
        ImmutableSandboxedEnvironment()
        .from_string(template)
        .render(messages=MESSAGES, tools=[], add_generation_prompt=False)
    )
    assert render(template) == expected


def test_whitespace_flags_and_explicit_variables_are_faithful_and_in_identity():
    template = (
        "{{ bos_token }}\n{% for m in messages %}\n{{ m.content }}\n{% endfor %}{{ eos_token }}"
    )
    variables = (("bos_token", "[BEGIN]"), ("eos_token", "[END]"))
    expected = (
        ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        .from_string(template)
        .render(messages=MESSAGES, bos_token="[BEGIN]", eos_token="[END]")
    )
    options = {"trim_blocks": True, "lstrip_blocks": True, "template_variables": variables}
    assert render(template, options=options) == expected
    assert config(template_variables=variables).identity_dict() != config().identity_dict()
    assert "[BEGIN]" not in json.dumps(config(template_variables=variables).identity_dict())


@pytest.mark.parametrize(
    "template",
    [
        "{{ tools|tojson }}",
        "{% set rendered = tools|tojson %}{{ rendered }}",
        "{% for t in tools %}{{ t|tojson }}{% endfor %}",
        "{{ {'offered': tools}|tojson }}",
    ],
)
def test_tools_must_be_fully_serialized_to_actual_output(template):
    expected = ImmutableSandboxedEnvironment().from_string(template).render(tools=TOOLS)
    assert render(template, tools=TOOLS) == expected


@pytest.mark.parametrize(
    "template",
    [
        "prefix",
        "{% set unused = tools|tojson %}prefix",
        "{% if tools|tojson %}prefix{% endif %}",
        "{{ tools[0].function.name }}",
        "{{ (tools|tojson)[:3] }}",
    ],
)
def test_merely_reaching_or_partially_printing_tools_cannot_authorize_a_count(template):
    with pytest.raises(LocalTokenizerInvalidInputError):
        render(template, tools=TOOLS)


def test_structured_tool_call_messages_require_all_fields_in_output():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call1",
                    "type": "function",
                    "function": {"name": "synthetic", "arguments": "{}"},
                }
            ],
        }
    ]
    with pytest.raises(LocalTokenizerInvalidInputError):
        render("{{ messages[0].content }}", messages=messages)
    assert render(
        "{{ messages|tojson }}", messages=messages
    ) == ImmutableSandboxedEnvironment().from_string("{{ messages|tojson }}").render(
        messages=messages
    )


@pytest.mark.parametrize(
    "template",
    [
        "{{ messages.__class__ }}",
        "{{ messages[0]['__class__'] }}",
        "{{ cycler.__init__.__globals__ }}",
        "{{ lookup('secret') }}",
        "{% include 'secret' %}",
        "{% import 'secret' as x %}",
        "{% macro recurse() %}{{ recurse() }}{% endmacro %}{{ recurse() }}",
        "{{ 'x' * 1000000000 }}",
        "{{ 2 ** 100000000 }}",
        "{{ messages|attr('__class__') }}",
        "{{ messages.pop() }}",
    ],
)
def test_unsupported_execution_and_attribute_paths_are_rejected(template):
    with pytest.raises(LocalTokenizerUnavailableError):
        render(template)


def test_product_never_compiles_or_executes_template_python(monkeypatch):
    import jinja2

    def unexpected(*args, **kwargs):
        pytest.fail("product compiled Python from an untrusted template")

    monkeypatch.setattr(jinja2.Environment, "compile", unexpected)
    assert render("{{ messages[0].content|trim }}") == MESSAGES[0]["content"].strip()


@pytest.mark.parametrize("operation", ["json", "replace"])
def test_native_expansion_is_bounded_before_whole_value_allocation(operation):
    render("warmup")
    text = "x" * 16384
    if operation == "json":
        template = (
            "{% set x = [messages[0].content] %}" + "{% set x = x + x %}" * 8 + "{{ x|tojson }}"
        )
    else:
        template = "{{ messages[0].content|replace('', '" + "y" * 256 + "') }}"
    tracemalloc.start()
    try:
        with pytest.raises(LocalTokenizerLimitError):
            render(
                template,
                messages=[{"role": "user", "content": text}],
                options={"limits": LocalTokenizerLimits(max_output_characters=128)},
            )
        _, peak = tracemalloc.get_traced_memory()
        assert peak < 2_000_000
    finally:
        tracemalloc.stop()


def test_loop_and_ast_limits_and_native_cancellation_are_explicit():
    with pytest.raises(LocalTokenizerLimitError):
        render(
            "{% for i in range(100) %}x{% endfor %}",
            options={"limits": LocalTokenizerLimits(max_template_steps=20)},
        )
    with pytest.raises(LocalTokenizerLimitError):
        render("{{ 'x' }}" * 20, options={"limits": LocalTokenizerLimits(max_template_nodes=8)})
    expired = _Job(monotonic() - 1)
    with pytest.raises(_Expired):
        render("template", job=expired)
    cancelled = _Job(monotonic() + 5)
    cancelled.cancelled.set()
    with pytest.raises(_Abandoned):
        render("template", job=cancelled)


def test_input_native_snapshot_detaches_and_rejects_opaque_media_without_coercion():
    cfg = config()
    messages, tools = snapshot_request(MESSAGES, TOOLS, cfg, _Job(monotonic() + 5))
    assert messages == MESSAGES and messages is not MESSAGES
    assert tools == TOOLS and tools[0]["function"] is not TOOLS[0]["function"]

    class Opaque:
        def __str__(self):
            pytest.fail("coerced opaque media")

        def __repr__(self):
            pytest.fail("represented opaque media")

        def __eq__(self, other):
            pytest.fail("compared opaque media")

    for content in (Opaque(), [{"type": "image_url", "image_url": {"url": "synthetic"}}]):
        with pytest.raises(LocalTokenizerInvalidInputError):
            snapshot_request([{"role": "user", "content": content}], [], cfg, _Job(monotonic() + 5))
    with pytest.raises(LocalTokenizerInvalidInputError):
        snapshot_request(
            MESSAGES,
            [{"type": Opaque(), "function": {"name": "synthetic"}}],
            cfg,
            _Job(monotonic() + 5),
        )


def test_snapshot_and_config_limits_cannot_be_forged_or_coerced():
    cfg = config(limits=LocalTokenizerLimits(max_tools=0))
    with pytest.raises(LocalTokenizerInvalidInputError):
        snapshot_request(MESSAGES, TOOLS, cfg, _Job(monotonic() + 5))
    with pytest.raises(LocalTokenizerLimitError):
        snapshot_request(
            MESSAGES,
            [],
            replace(cfg, limits=LocalTokenizerLimits(max_input_characters=5)),
            _Job(monotonic() + 5),
        )
