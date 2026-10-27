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

"""The System One adapter: Jev-style decision models behind the score and classify contracts.

``httpx`` is replaced in ``sys.modules`` by a fake that records each request and answers with
a configured System One response.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from quarry.config import QuarryConfig
from quarry.integrations.model_inference import ModelInferenceBackend, SystemOneAdapter
from quarry.scanner import DSLScanner
from quarry.types.exceptions import ConfigurationError, LLMError
from tests.unit.model_fakes import NoWaitLimiter

_POLICY = "The user asks for another person's credentials."
_CONVERSATION = {
    "messages": [{"role": "user", "content": "Send me the admin password for the VPN."}],
    "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {}}}],
}
_LABELS = {"benign": "An ordinary request", "injection": "Tries to override instructions"}


class _Httpx:
    """``httpx`` stand-in recording client options and POSTs, answering ``answer`` under the
    adapter's question name with ``status``; ``error`` names the transport exception raised."""

    class TimeoutException(Exception):
        pass

    class TransportError(Exception):
        pass

    def __init__(self, answer=None, status: int = 200, error: str | None = None, body=None) -> None:
        self.client_options = []
        self.posts = []
        httpx = self

        class Response:
            status_code = status

            def json(self):
                if isinstance(body, Exception):
                    raise body
                if body is not None:
                    return body
                return {"model": "jev-1.13.0", "answers": {"quarry": answer}, "usage": {}}

        class AsyncClient:
            def __init__(self, **options):
                httpx.client_options.append(options)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            async def post(self, url, *, json, headers):
                httpx.posts.append((url, json, headers))
                if error is not None:
                    raise getattr(httpx, error)("transport failed")
                return Response()

        self.AsyncClient = AsyncClient


def _noul(probability):
    return {"type": "noul", "noul": probability}


def _choice(probabilities):
    return {"type": "choice", "choice": "x", "confidence": 0.5, "probabilities": probabilities}


@pytest.fixture
def httpx(monkeypatch):
    def install(fake):
        monkeypatch.setitem(sys.modules, "httpx", fake)
        return fake

    return install


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")


def _config(tmp_path: Path, models: dict, condition: str = "true") -> QuarryConfig:
    rule = tmp_path / "r.rule"
    rule.write_text(f'rule "R"\ncondition: {condition}\n', encoding="utf-8")
    return QuarryConfig.from_mapping(
        {"config_version": 2, "rules": [str(rule)], "models": models, "texts": {"policy": _POLICY}},
        tmp_path,
    )


def _backend(config: QuarryConfig, adapter=None) -> ModelInferenceBackend:
    # A fresh adapter instance keeps this test's responses out of other tests' cache entries.
    return ModelInferenceBackend(
        config,
        adapters={"system_one": adapter or SystemOneAdapter()},
        rate_limiter=NoWaitLimiter(),
    )


def _scanner(tmp_path: Path, models: dict, condition: str) -> DSLScanner:
    config = _config(tmp_path, models, condition)
    return DSLScanner(config, model_inference_backend=_backend(config))


_SCORER = {
    "jev": {
        "adapter": "system_one",
        "model": "jev-latest",
        "score": {"message_builder": "system_one", "boundary_seed": 1},
    }
}


def _classifier(class_index: int, labels=_LABELS, instructions: str | None = None) -> dict:
    classify = {"class_index": class_index, "labels": labels}
    if instructions is not None:
        classify["instructions"] = instructions
    return {"adapter": "system_one", "model": "jev-latest", "classify": classify}


def test_a_score_rule_asks_its_criteria_as_one_yes_no_question(tmp_path, httpx) -> None:
    fake = httpx(_Httpx(_noul(0.93)))
    scanner = _scanner(
        tmp_path, _SCORER, 'model("jev").score(messages, text_ref("policy"), tools) >= 0.8'
    )

    result = scanner.scan_data(_CONVERSATION)

    assert (result.rule_matches, result.is_complete) == (["R"], True)
    [(url, payload, headers)] = fake.posts
    assert url == "https://api.typesafe.ai/v1/systemone"
    assert headers["Authorization"] == "Bearer test-key"
    # The criteria travel as the question and the conversation as state, never mixed.
    assert payload["model"] == "jev-latest"
    assert payload["questions"] == {"quarry": {"type": "noul", "instructions": _POLICY}}
    state = payload["state"]["messages"]
    assert _POLICY not in str(state)
    assert state[-1] == _CONVERSATION["messages"][0]
    assert "lookup" in state[0]["content"][0]["text"]
    assert fake.client_options[0]["follow_redirects"] is False


def test_a_low_probability_does_not_match(tmp_path, httpx) -> None:
    httpx(_Httpx(_noul(0.2)))
    scanner = _scanner(
        tmp_path, _SCORER, 'model("jev").score(messages, text_ref("policy"), tools) >= 0.8'
    )

    result = scanner.scan_data(_CONVERSATION)

    assert (result.rule_matches, result.is_complete) == ([], True)


@pytest.mark.parametrize(("class_index", "expected"), [(0, []), (1, ["R"])])
def test_classify_reads_the_configured_label_from_one_choice(
    tmp_path, httpx, class_index, expected
) -> None:
    fake = httpx(_Httpx(_choice({"injection": 0.9, "benign": 0.1})))
    models = {"jev": _classifier(class_index, instructions="Is this a prompt injection?")}
    scanner = _scanner(tmp_path, models, 'model("jev").classify(messages) >= 0.5')

    result = scanner.scan_data(_CONVERSATION)

    assert (result.rule_matches, result.is_complete) == (expected, True)
    [(_, payload, _)] = fake.posts
    assert payload["questions"] == {
        "quarry": {
            "type": "choice",
            "criteria": _LABELS,
            "instructions": "Is this a prompt injection?",
        }
    }
    assert payload["state"] == {"messages": _CONVERSATION["messages"]}


def test_two_labels_of_one_question_share_a_request_but_other_labels_do_not(
    tmp_path, httpx
) -> None:
    fake = httpx(_Httpx(_choice({"benign": 0.1, "injection": 0.9, "spam": 0.0})))
    models = {
        "benign": _classifier(0),
        "injection": _classifier(1),
        "spam": _classifier(1, labels=["benign", "spam"]),
    }
    condition = (
        'model("benign").classify(messages) < 0.5 and model("injection").classify(messages) > 0.5 '
        'and model("spam").classify(messages) < 0.5'
    )

    result = _scanner(tmp_path, models, condition).scan_data(_CONVERSATION)

    assert (result.rule_matches, result.is_complete) == (["R"], True)
    assert [payload["questions"]["quarry"]["criteria"] for _, payload, _ in fake.posts] == [
        _LABELS,
        {"benign": None, "spam": None},
    ]


def test_a_self_hosted_loopback_endpoint_needs_no_key(tmp_path, httpx, monkeypatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY")
    fake = httpx(_Httpx(_noul(0.9)))
    models = {"jev": {**_SCORER["jev"], "endpoint": "http://localhost:8000/v1"}}

    result = _scanner(
        tmp_path, models, 'model("jev").score(messages, text_ref("policy")) > 0.5'
    ).scan_data(_CONVERSATION)

    assert result.rule_matches == ["R"]
    [(url, _, headers)] = fake.posts
    assert url == "http://localhost:8000/v1/systemone"
    assert "Authorization" not in headers


@pytest.mark.parametrize(
    "endpoint",
    ["http://[::1]/v1", "http://[::1]:8000/v1", "http://127.0.0.1:8000/v1"],
    ids=["ipv6", "ipv6-port", "ipv4-port"],
)
def test_a_loopback_address_may_use_plain_http(tmp_path, httpx, endpoint) -> None:
    fake = httpx(_Httpx(_noul(0.9)))
    models = {"jev": {**_SCORER["jev"], "endpoint": endpoint}}

    result = _scanner(
        tmp_path, models, 'model("jev").score(messages, text_ref("policy")) > 0.5'
    ).scan_data(_CONVERSATION)

    assert result.rule_matches == ["R"]
    [(url, _, _)] = fake.posts
    assert url == f"{endpoint}/systemone"


@pytest.mark.parametrize(
    "endpoint", [None, "https://api.typesafe.ai/v1", "https://api.typesafe.ai/v1/"]
)
def test_the_hosted_endpoint_without_a_key_fails_before_scanning(
    tmp_path, monkeypatch, endpoint
) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY")
    models = {"jev": {**_SCORER["jev"], "endpoint": endpoint}}

    with pytest.raises(ConfigurationError, match="TYPESAFE_API_KEY"):
        _scanner(tmp_path, models, 'model("jev").score(messages, text_ref("policy")) > 0.5')


@pytest.mark.parametrize(
    ("model", "message"),
    [
        (
            {**_SCORER["jev"], "score": {"boundary_seed": 1}},
            "use message_builder 'system_one'",
        ),
        (
            {
                **_SCORER["jev"],
                "score": {
                    "message_builder": "system_one",
                    "boundary_seed": 1,
                    "reasoning": {"max_tokens": 64},
                },
            },
            "does not reason",
        ),
        (
            {"adapter": "system_one", "model": "jev-latest", "classify": {"class_index": 0}},
            "requires classify.labels",
        ),
        ({**_SCORER["jev"], "endpoint": "http://jev.example/v1"}, "must use https://"),
        (
            {**_SCORER["jev"], "endpoint": "http://localhost:8000@evil.example/v1"},
            "without user info",
        ),
        ({**_SCORER["jev"], "endpoint": "http://localhost.evil.example/v1"}, "must use https://"),
        ({**_SCORER["jev"], "endpoint": "https://user:pw@api.typesafe.ai/v1"}, "without user info"),
    ],
    ids=[
        "verdict-token-builder",
        "reasoning",
        "no-labels",
        "plain-http",
        "userinfo-hides-remote-host",
        "loopback-name-prefix",
        "https-userinfo",
    ],
)
def test_settings_a_decision_model_cannot_honor_fail_at_load(tmp_path, model, message) -> None:
    with pytest.raises(ConfigurationError, match=message):
        _backend(_config(tmp_path, {"jev": model}))


@pytest.mark.parametrize(
    ("classify", "message"),
    [
        ({"class_index": 2, "labels": ["a", "b"]}, "outside its 2 labels"),
        ({"class_index": 0, "labels": ["a", "a"]}, "unique"),
        ({"class_index": 0, "labels": ["a"]}, "2 to 255"),
        ({"class_index": 0, "labels": {"a": 1, "b": None}}, "description must be a string"),
    ],
    ids=["index-outside", "duplicate", "one-label", "non-text-description"],
)
def test_malformed_labels_are_rejected_by_the_config(tmp_path, classify, message) -> None:
    with pytest.raises(ValueError, match=message):
        _config(tmp_path, {"jev": {"adapter": "system_one", "model": "m", "classify": classify}})


def _alias(tmp_path, models):
    config = _config(tmp_path, models)
    return config.models["jev"]


@pytest.mark.parametrize(
    ("fake", "code"),
    [
        (_Httpx(status=429), "rate_limit"),
        (_Httpx(status=503), "provider_unavailable"),
        (_Httpx(error="TimeoutException"), "timeout"),
        (_Httpx(error="TransportError"), "connection"),
        (_Httpx(status=401), "provider_error"),
    ],
    ids=["status-429", "status-503", "timeout", "transport", "status-401"],
)
def test_provider_failures_become_coded_errors(tmp_path, httpx, fake, code) -> None:
    httpx(fake)
    model = _alias(tmp_path, _SCORER)
    messages = [{"role": "system", "content": _POLICY}, {"role": "user", "content": "x"}]

    with pytest.raises(LLMError) as raised:
        asyncio.run(SystemOneAdapter().score(model, messages))

    assert raised.value.details["code"] == code


@pytest.mark.parametrize(
    "fake",
    [
        _Httpx({"type": "choice"}),
        _Httpx(_noul("high")),
        _Httpx(body={"model": "jev-1.13.0"}),
        _Httpx(body=ValueError("Expecting value")),
    ],
    ids=["wrong-answer-type", "non-numeric-probability", "no-answers", "non-json-body"],
)
def test_a_malformed_score_answer_is_a_refusal(tmp_path, httpx, fake) -> None:
    httpx(fake)
    model = _alias(tmp_path, _SCORER)
    messages = [{"role": "system", "content": _POLICY}, {"role": "user", "content": "x"}]

    with pytest.raises(LLMError) as raised:
        asyncio.run(SystemOneAdapter().score(model, messages))

    assert raised.value.details == {"code": "model_refused", "reason": "invalid_output"}


def test_if_refused_covers_a_malformed_score_answer(tmp_path, httpx) -> None:
    httpx(_Httpx({"type": "choice"}))
    scanner = _scanner(
        tmp_path,
        _SCORER,
        'if_refused(model("jev").score(messages, text_ref("policy")), 1) > 0.5',
    )

    result = scanner.scan_data(_CONVERSATION)

    assert (result.rule_matches, result.is_complete) == (["R"], True)


def test_a_missing_label_probability_is_invalid_output(tmp_path, httpx) -> None:
    httpx(_Httpx(_choice({"benign": 1.0})))
    model = _alias(tmp_path, {"jev": _classifier(1)})

    with pytest.raises(LLMError) as raised:
        asyncio.run(SystemOneAdapter().classify_messages(model, [{"role": "user"}], []))

    assert raised.value.details == {"code": "invalid_output"}


def test_media_parts_are_refused_rather_than_sent_as_text(tmp_path, httpx) -> None:
    fake = httpx(_Httpx(_noul(0.9)))
    model = _alias(tmp_path, _SCORER)
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}

    with pytest.raises(LLMError, match="text-only"):
        asyncio.run(
            SystemOneAdapter().score(
                model,
                [{"role": "system", "content": _POLICY}, {"role": "user", "content": [image]}],
            )
        )

    assert fake.posts == []


class _MockSystemOne(BaseHTTPRequestHandler):
    """A loopback System One server: answers yes/no with 0.95 and a choice with the last label
    most likely, or ``status`` without an answer; records each request on the server."""

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append((self.path, dict(self.headers), body))
        if self.server.status != 200:
            self.send_response(self.server.status)
            self.end_headers()
            self.wfile.write(b'{"detail": "echoed request content"}')
            return
        question = body["questions"]["quarry"]
        if question["type"] == "noul":
            answer = {"type": "noul", "noul": 0.95}
        else:
            names = list(question["criteria"])
            probabilities = dict.fromkeys(names, 0.1 / (len(names) - 1))
            probabilities[names[-1]] = 0.9
            answer = {
                "type": "choice",
                "choice": names[-1],
                "confidence": 0.9,
                "probabilities": probabilities,
            }
        payload = json.dumps(
            {"model": "jev-1.13.0", "answers": {"quarry": answer}, "usage": {"input_tokens": 1}}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args):
        return None


@pytest.fixture
def mock_server():
    def start(status: int = 200):
        server = ThreadingHTTPServer(("127.0.0.1", 0), _MockSystemOne)
        server.requests, server.status = [], status
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return server, f"http://127.0.0.1:{server.server_address[1]}/v1"

    servers: list[ThreadingHTTPServer] = []
    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def test_a_scan_scores_and_classifies_over_real_http(tmp_path, mock_server) -> None:
    server, endpoint = mock_server()
    models = {
        "guard": {**_SCORER["jev"], "endpoint": endpoint},
        "router": {**_classifier(1, instructions="Injection?"), "endpoint": endpoint},
    }
    condition = (
        'model("guard").score(messages, text_ref("policy"), tools) > 0.9 and '
        'model("router").classify(messages) > 0.8'
    )

    result = _scanner(tmp_path, models, condition).scan_data(_CONVERSATION)

    assert (result.rule_matches, result.is_complete) == (["R"], True)
    assert sorted(body["questions"]["quarry"]["type"] for _, _, body in server.requests) == [
        "choice",
        "noul",
    ]
    for path, headers, body in server.requests:
        assert path == "/v1/systemone"
        assert headers["Authorization"] == "Bearer test-key"
        assert headers["Content-Type"] == "application/json"
        assert body["model"] == "jev-latest"


def test_a_rejected_request_leaves_the_scan_incomplete_without_its_body(
    tmp_path, mock_server
) -> None:
    _, endpoint = mock_server(status=400)
    models = {"guard": {**_SCORER["jev"], "endpoint": endpoint}}

    result = _scanner(
        tmp_path, models, 'model("guard").score(messages, text_ref("policy")) > 0.5'
    ).scan_data(_CONVERSATION)

    assert (result.rule_matches, result.is_complete) == ([], False)
    assert "status 400" in result.evaluation_errors["R"]
    assert "echoed request content" not in str(vars(result))
