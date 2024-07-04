#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/06/28 10:44:41
@Author  :   ChenHao
@Description  :
@Contact :   jerrychen1990@gmail.com
'''

from http import HTTPStatus
from typing import Any, List, Tuple

from loguru import logger

from liteai.core import BaseProvider, ModelResponse, Message, Usage

import dashscope


class QwenProvider(BaseProvider):
    key: str = "qwen"
    allow_kwargs = {"do_sample", "stream", "temperature", "top_p", "max_tokens"}
    api_key_env = "DASHSCOPE_API_KEY"

    def __init__(self, api_key: str = None):
        super().__init__(api_key=api_key)
        dashscope.api_key = self.api_key

    def pre_process(self, model: str, messages: List[Message], stream: bool, **kwargs) -> Tuple[List[dict], dict]:
        messages, kwargs = super().pre_process(model, messages, stream, **kwargs)
        if stream:
            kwargs["incremental_output"] = True
        for message in messages:
            # logger.debug(f"{message=}")
            if message.get("image"):
                message["content"] = [dict(text=message["content"]),
                                      dict(image=message["image"])]
                del message["image"]
        return messages, kwargs

    def _inner_complete_(self, model, messages: List[dict], stream: bool, ** kwargs) -> Any:
        if "vl" in model:
            response = dashscope.MultiModalConversation.call(model=model,
                                                             messages=messages,
                                                             **kwargs)
        else:
            response = dashscope.Generation.call(model=model,
                                                 messages=messages,
                                                 stream=stream,
                                                 result_format="message",
                                                 **kwargs)
        return response

    def post_process(self, response: dict) -> ModelResponse:
        logger.debug(f"{response=}, {type(response)=}")
        if response.status_code == HTTPStatus.OK:
            content = response.output.choices[0].message.content
            usage = Usage(prompt_tokens=response.usage.input_tokens, completion_tokens=response.usage.output_tokens)
            return ModelResponse(content=content, usage=usage)
        else:
            raise Exception(f"{response.status_code} {response.message}")

    def post_process_stream(self, response) -> ModelResponse:
        def _gen():
            acc = []
            for chunk in response:
                # logger.debug(f"{chunk=}")
                if chunk.status_code == HTTPStatus.OK:
                    choices = chunk.output.choices
                    if choices:
                        choice = choices[0]
                        if choice.message.content:
                            delta_content = choice.message.content
                            # logger.info(f"{delta_content}")
                            yield delta_content
                            acc.append(delta_content)
                else:
                    logger.error(f"{chunk.status_code} {chunk.message}")

            resp_msg = "".join(acc).strip()
            logger.debug(f"model generate answer:{resp_msg}")
        return ModelResponse(content=_gen())


if __name__ == "__main__":
    provider = QwenProvider()
    messages = [Message(role="user", content="你好")]
    resp = provider.complete(messages=messages, model="glm-3-turbo", stream=False)
    print(resp.content)
