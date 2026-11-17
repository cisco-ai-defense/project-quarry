# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Host error privacy and ownership of copied media-derived content."""

import pytest

from quarry.types.exceptions import LLMError, ModelError
from quarry.types.messages import Messages
from tests.unit.test_adapter_transform_dsl import Adapter, compile_condition, configured


@pytest.mark.parametrize("model", [False, True])
@pytest.mark.parametrize("exception", [ModelError, LLMError])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_adapter_typed_errors_do_not_publish_host_text_or_details(
    tmp_path, model, exception, asynchronous
):
    marker = "SYNTHETIC_PRIVATE_BODY"

    def fail(*args, **kwargs):
        raise exception(marker, {"transform": "convert", "code": marker, "payload": marker})

    async def fail_async(*args, **kwargs):
        fail()

    adapter = Adapter()
    adapter.transform = fail_async if asynchronous else fail
    _, backend, _ = configured(tmp_path, adapter=adapter, model=model)
    with pytest.raises((ModelError, LLMError)) as error:
        await backend.transform_unit("convert", {"type": "text", "text": "owned"})
    assert marker not in str(error.value)
    assert marker not in repr(error.value.details)


def test_copied_media_text_cannot_be_rebound_to_a_foreign_conversation(tmp_path):
    config, backend, _ = configured(tmp_path)
    original = backend.model_classify_messages
    captured = []

    async def capture(alias, messages, *args, **kwargs):
        captured.append(messages)
        return await original(alias, messages, *args, **kwargs)

    backend.model_classify_messages = capture
    policy = compile_condition(
        config,
        backend,
        'model("judge").classify(message("user", messages.transform("convert")[0].content) '
        '+ message("assistant", "review")) > 0.5',
    )
    result = policy.scan(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
                    ],
                }
            ]
        }
    )
    assert result.is_complete and result.rule_matches == ["R"]
    copied = captured[0]
    assert copied.to_openai_messages() == [
        {"role": "user", "content": "LABEL"},
        {"role": "assistant", "content": "review"},
    ]
    foreign = Messages.from_openai([{"role": "user", "content": "unrelated"}])
    with pytest.raises(ValueError, match="canonical source"):
        Messages.combine((copied, foreign))


@pytest.mark.parametrize("exception", [ModelError, LLMError])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_whole_message_typed_errors_do_not_publish_host_payloads(
    tmp_path, exception, asynchronous
):
    marker = "SYNTHETIC_PRIVATE_WHOLE_MESSAGE"

    class Whole:
        local = True
        transform_parts = ()

        def transform_messages(self, config, messages):
            if asynchronous:
                return fail_async()
            raise exception(marker, {"payload": marker})

    async def fail_async():
        raise exception(marker, {"payload": marker})

    _, backend, _ = configured(tmp_path, adapter=Whole())
    with pytest.raises((ModelError, LLMError)) as error:
        await backend.transform_messages(
            "convert", Messages.from_openai([{"role": "user", "content": "owned"}])
        )
    assert marker not in str(error.value) and marker not in repr(error.value.details)
