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

"""Semantic probers: the fail-closed load gate and the dependency-free stub backends.

``off_topic`` / ``topic_drift`` / ``capability_tags`` / ``unrequested_action`` ground on a
registered prober backend. The keyword and lexical stubs are crude, so a rule using a prober
fails to load unless a backend is registered or the config opts in with ``allow_stub_probers``.
The DSL-level drift signatures are exercised in ``tests/engine/test_corpus.py``.
"""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import pytest

from quarry.config import QuarryConfig
from quarry.integrations.resources import ResourcePreparationError
from quarry.probers import CapabilityClassifier, TopicProber
from quarry.scanner import DSLScanner
from quarry.types.exceptions import IncompleteScanError
from tests.conftest import scanner_from_rules

_TOPIC_RULE = 'rule "Topic" condition: any m in assistant_messages where off_topic(m.content)'
_CAPABILITY_RULE = (
    'rule "Capability" condition: any c in tool_calls where '
    'capability_tags("code_execution", c.name)'
)
_INTENT = "please summarize the billing support ticket"
_THREE_DETECTORS = """rule "Reject unsafe action"
condition:
    any call in tool_calls where (
        off_topic(call.name)
        or capability_tags("state_modification", call.name)
        or unrequested_action(call.name)
    )
"""
_DRIFT_RULE = 'rule "Drift" condition: any c in tool_calls where topic_drift(c.name) > 0.5'


@pytest.mark.parametrize(
    ("rule", "named"),
    [(_TOPIC_RULE, "topic prober"), (_CAPABILITY_RULE, "capability classifier")],
    ids=["topic", "capability"],
)
def test_a_prober_rule_without_a_backend_fails_to_load(tmp_path, rule, named) -> None:
    with pytest.raises(ResourcePreparationError, match=f"no {named} backend is registered"):
        scanner_from_rules(tmp_path, rule)


def _turn(reply: str, tool: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": _INTENT},
            {
                "role": "assistant",
                "content": reply,
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": tool, "arguments": "{}"}}
                ],
            },
        ]
    }


class _FixedTopicReply:
    def __init__(self, value: object) -> None:
        self.value = value

    def off_topic(self, intent: str, action: str) -> object:
        assert (intent, action) == (_INTENT, "delete_account")
        return self.value


class _FixedDriftReply:
    def __init__(self, value: object) -> None:
        self.value = value

    def drift_score(self, intent: str, action: str) -> object:
        assert (intent, action) == (_INTENT, "delete_account")
        return self.value


class _FixedCapabilityReply:
    def __init__(self, tag: object, unrequested: object) -> None:
        self.tag = tag
        self.unrequested = unrequested

    def has_tag(self, action: str, tag: str) -> object:
        assert (action, tag) == ("delete_account", "state_modification")
        return self.tag

    def unrequested_capability(self, action: str, intent: str) -> object:
        assert (action, intent) == ("delete_account", _INTENT)
        return self.unrequested


def _scan_three_detectors(topic: object, tag: object, unrequested: object):
    scanner = DSLScanner(
        QuarryConfig(
            config_version=2,
            quarry_rule_texts=(_THREE_DETECTORS,),
            allow_stub_probers=True,
        )
    )
    scanner.resource_manager.topic_prober = _FixedTopicReply(topic)
    scanner.resource_manager.capability_classifier = _FixedCapabilityReply(tag, unrequested)
    return scanner.scan_data(_turn("", "delete_account"))


def _scan_drift(value: object, rule: str = _DRIFT_RULE):
    scanner = DSLScanner(
        QuarryConfig(
            config_version=2,
            quarry_rule_texts=(rule,),
            allow_stub_probers=True,
        )
    )
    scanner.resource_manager.topic_prober = _FixedDriftReply(value)
    return scanner.scan_data(_turn("", "delete_account"))


@pytest.mark.parametrize(
    ("score", "matches"),
    [
        (0, []),
        (0.0, []),
        (0.5, []),
        (Decimal("0.5"), []),
        (Fraction(1, 2), []),
        (1.0, ["Drift"]),
        (Decimal("1"), ["Drift"]),
        (1, ["Drift"]),
    ],
)
def test_valid_drift_scores_keep_their_verdicts(
    score: int | float | Decimal | Fraction, matches: list[str]
) -> None:
    result = _scan_drift(score)

    assert (result.rule_matches, result.completion_state) == (matches, "complete")


@pytest.mark.parametrize(
    "score",
    [
        Decimal("0.50000000000000000001"),
        Fraction(2**54 + 1, 2**55),
    ],
    ids=["decimal", "fraction"],
)
def test_precise_drift_scores_keep_their_threshold_semantics(
    score: Decimal | Fraction,
) -> None:
    result = _scan_drift(score)

    assert (result.rule_matches, result.completion_state) == (["Drift"], "complete")


def test_a_small_positive_drift_score_does_not_underflow_to_zero() -> None:
    rule = 'rule "Drift" condition: any c in tool_calls where topic_drift(c.name) > 0'
    result = _scan_drift(Decimal("1E-10000"), rule)

    assert (result.rule_matches, result.completion_state) == (["Drift"], "complete")


@pytest.mark.parametrize(
    "score",
    [
        False,
        -1.0,
        1.1,
        Fraction(-1, 2**1075),
        Fraction(2**54 + 1, 2**54),
        Decimal("-1E-10000"),
        Decimal("1.00000000000000000001"),
        Decimal("NaN"),
        Decimal("Infinity"),
        float("nan"),
        float("inf"),
        float("-inf"),
        None,
        "0.9",
    ],
    ids=[
        "bool",
        "negative",
        "above-one",
        "fraction-rounds-to-negative-zero",
        "fraction-rounds-to-one",
        "decimal-rounds-to-negative-zero",
        "decimal-rounds-to-one",
        "decimal-nan",
        "decimal-inf",
        "nan",
        "inf",
        "negative-inf",
        "none",
        "str",
    ],
)
def test_malformed_drift_score_fails_the_rule(score: object) -> None:
    result = _scan_drift(score)

    assert (result.rule_matches, result.completion_state) == ([], "incomplete")
    assert "expected a finite non-boolean number in [0, 1]" in result.evaluation_errors["Drift"]
    with pytest.raises(IncompleteScanError):
        result.raise_for_status()


def test_nan_drift_cannot_complete_a_less_than_policy() -> None:
    rule = 'rule "Drift" condition: any c in tool_calls where topic_drift(c.name) < 0.5'
    result = _scan_drift(float("nan"), rule)

    assert (result.rule_matches, result.completion_state) == ([], "incomplete")
    assert "expected a finite non-boolean number in [0, 1]" in result.evaluation_errors["Drift"]


@pytest.mark.parametrize(
    ("topic", "tag", "unrequested", "matches"),
    [
        (False, False, False, []),
        (True, False, False, ["Reject unsafe action"]),
        (False, True, False, ["Reject unsafe action"]),
        (False, False, True, ["Reject unsafe action"]),
    ],
)
def test_valid_boolean_semantic_replies_keep_their_verdicts(
    topic: bool, tag: bool, unrequested: bool, matches: list[str]
) -> None:
    result = _scan_three_detectors(topic, tag, unrequested)

    assert (result.rule_matches, result.completion_state) == (matches, "complete")


@pytest.mark.parametrize("value", [None, 0, [], 1], ids=["none", "zero", "empty", "one"])
@pytest.mark.parametrize("detector", ["off_topic", "capability_tags", "unrequested_action"])
def test_malformed_semantic_reply_fails_the_rule(detector: str, value: object) -> None:
    replies = {
        "off_topic": False,
        "capability_tags": False,
        "unrequested_action": False,
    }
    replies[detector] = value

    result = _scan_three_detectors(
        replies["off_topic"], replies["capability_tags"], replies["unrequested_action"]
    )

    assert (result.rule_matches, result.completion_state) == ([], "incomplete")
    assert "expected bool" in result.evaluation_errors["Reject unsafe action"]
    with pytest.raises(IncompleteScanError):
        result.raise_for_status()


def test_three_falsey_malformed_replies_cannot_bypass_redundant_policy() -> None:
    result = _scan_three_detectors(None, None, None)

    assert (result.rule_matches, result.completion_state) == ([], "incomplete")
    assert "expected bool" in result.evaluation_errors["Reject unsafe action"]


@pytest.mark.parametrize(
    ("conversation", "matches"),
    [
        (
            _turn("buying bitcoin ethereum crypto wallet mining rig now", "execute_code"),
            ["Topic", "Capability"],
        ),
        (_turn("here is the billing support ticket summary", "get_ticket"), []),
    ],
    ids=["drifting-turn", "on-topic-turn"],
)
def test_opting_into_the_stubs_loads_and_grounds_prober_rules(
    tmp_path, conversation, matches
) -> None:
    scanner = scanner_from_rules(
        tmp_path, _TOPIC_RULE, _CAPABILITY_RULE, config_overrides={"allow_stub_probers": True}
    )

    result = scanner.scan_data(conversation)

    assert (sorted(result.rule_matches), result.is_complete) == (sorted(matches), True)


@pytest.mark.parametrize(
    ("prober", "intent", "action", "off_topic"),
    [
        (TopicProber(), _INTENT, "here is the billing support ticket summary", False),
        (TopicProber(), _INTENT, "buying bitcoin ethereum crypto wallet mining rig now", True),
        # No baseline intent: never claim drift.
        (TopicProber(), "", "anything at all", False),
        (
            TopicProber(off_topic_threshold=0.1),
            "summarize the ticket",
            "summarize a slightly different ticket topic",
            True,
        ),
        (
            TopicProber(off_topic_threshold=0.99),
            "summarize the ticket",
            "summarize a slightly different ticket topic",
            False,
        ),
    ],
    ids=["on-topic", "off-topic", "no-intent", "strict-threshold", "lenient-threshold"],
)
def test_the_lexical_topic_stub(prober, intent, action, off_topic) -> None:
    assert prober.off_topic(intent, action) is off_topic


class _FixedEmbedder:
    """Bag-of-words vectors over a fixed vocabulary."""

    _VOCAB = ["billing", "ticket", "crypto", "wallet"]

    def embed(self, text: str) -> list[float]:
        return [float(text.lower().count(word)) for word in self._VOCAB]


@pytest.mark.parametrize(
    ("action", "drift"),
    [("billing ticket", 0.0), ("crypto wallet", 1.0)],
    ids=["same", "orthogonal"],
)
def test_a_registered_embedder_replaces_the_lexical_measure(action, drift) -> None:
    prober = TopicProber(embedder=_FixedEmbedder())

    assert prober.drift_score("billing ticket", action) == pytest.approx(drift, abs=1e-9)


@pytest.mark.parametrize(
    ("tool", "tag", "tagged"),
    [
        ("http_post", "external_communication", True),
        ("send_email", "external_communication", True),
        ("get_ticket", "external_communication", False),
        ("query_db", "private_data_access", True),
        ("read_secrets", "private_data_access", True),
        ("format_text", "private_data_access", False),
        ("delete_record", "state_modification", True),
        ("issue_refund", "state_modification", True),
        ("execute_code", "code_execution", True),
        ("fetch_url_and_http_post", "untrusted_ingress", True),
        ("fetch_url_and_http_post", "external_communication", True),
        ("", "external_communication", False),
        ("http_post", "not_a_real_tag", False),
    ],
)
def test_the_keyword_capability_stub(tool, tag, tagged) -> None:
    assert CapabilityClassifier().has_tag(tool, tag) is tagged


@pytest.mark.parametrize(
    ("tool", "intent", "unrequested"),
    [
        ("issue_refund", "summarize the support ticket", True),
        ("http_post", "summarize the support ticket", True),
        ("issue_refund", "please issue a refund for my order", False),
        ("get_ticket", "summarize the support ticket", False),
        ("query_db", "summarize the support ticket", False),
        # No intent licenses a consequential action; a read-only one is still not consequential.
        ("issue_refund", "", True),
        ("get_ticket", "", False),
    ],
)
def test_an_unrequested_capability_is_a_consequential_action_the_user_did_not_ask_for(
    tool, intent, unrequested
) -> None:
    assert CapabilityClassifier().unrequested_capability(tool, intent) is unrequested


def test_a_registered_classifier_replaces_the_keyword_map() -> None:
    class Classifier:
        def classify(self, text: str) -> set[str]:
            return {"code_execution"} if "run" in text else set()

    capabilities = CapabilityClassifier(classifier=Classifier())

    assert [
        capabilities.has_tag("run_thing", "code_execution"),
        capabilities.has_tag("run_thing", "external_communication"),
        capabilities.has_tag("http_post", "external_communication"),
    ] == [True, False, False]
