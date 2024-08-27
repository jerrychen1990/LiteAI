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

from liteai.core import ModelResponse, Message, ToolDesc, Usage
from liteai.provider.base import BaseProvider

import dashscope
from dashscope.api_entities.dashscope_response import GenerationResponse, MultiModalConversationResponse
from snippets import add_callback2gen

from liteai.utils import acc_chunks


class QwenProvider(BaseProvider):
    key: str = "qwen"
    allow_kwargs = {"stream", "temperature", "top_p", "max_tokens"}
    api_key_env = "DASHSCOPE_API_KEY"

    def __init__(self, api_key: str = None, **kwargs):
        super().__init__(api_key=api_key)
        dashscope.api_key = self.api_key

    def pre_process(self, model: str, messages: List[Message], tools: List[ToolDesc], stream: bool, **kwargs) -> Tuple[List[dict], dict]:
        messages, tools, kwargs = super().pre_process(model=model, messages=messages, tools=tools, stream=stream, **kwargs)
        if stream:
            kwargs["incremental_output"] = True
        for message in messages:
            # logger.debug(f"{message=}")
            if message.get("image"):
                message["content"] = [dict(text=message["content"]),
                                      dict(image=message["image"])]
                del message["image"]
        return messages, tools, kwargs

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

    def post_process(self, response: dict, **kwargs) -> ModelResponse:
        logger.debug(f"{response=}, {type(response)=}")
        if response.status_code == HTTPStatus.OK:
            if isinstance(response, GenerationResponse):
                content = response.output.choices[0].message.content
            elif isinstance(response, MultiModalConversationResponse):
                content = response.output.choices[0].message.content[0]["text"]
            usage = Usage(prompt_tokens=response.usage.input_tokens, completion_tokens=response.usage.output_tokens)
            return ModelResponse(content=content, usage=usage)
        else:
            raise Exception(f"{response.status_code} {response.message}")

    def post_process_stream(self, response, **kwargs) -> ModelResponse:
        def _handel_chunk(chunk):

            if chunk.status_code == HTTPStatus.OK:
                choices = chunk.output.choices
                if choices:
                    choice = choices[0]
                    if choice.message.content:
                        delta_content = choice.message.content
                        return delta_content

        gen = (e for e in (_handel_chunk(chunk) for chunk in response) if e)
        gen = add_callback2gen(gen, acc_chunks)
        return ModelResponse(content=gen)


if __name__ == "__main__":
    provider = QwenProvider()
    messages = [Message(role="user", content="你好")]
    resp = provider.complete(messages=messages, model="qwen-turbo", stream=False)
    print(resp.content)
