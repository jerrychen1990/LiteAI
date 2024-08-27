#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/07/31 15:13:08
@Author  :   ChenHao
@Description  :   openai接口
@Contact :   jerrychen1990@gmail.com
'''


from typing import Any, List, Tuple

from loguru import logger


from liteai.core import ModelCard, ModelResponse, Message, ToolDesc, Usage
from liteai.provider.base import BaseProvider
from openai import OpenAI
from snippets.utils import add_callback2gen

from liteai.utils import image2base64

from snippets import add_callback2gen

from liteai.utils import get_text_chunk, image2base64, acc_chunks


class OpenAIProvider(BaseProvider):
    key: str = "openai"
    allow_kwargs = {"stream", "temperature", "top_p", "max_tokens"}
    api_key_env = "OPENAI_API_KEY"

    def __init__(self, api_key: str = None, base_url: str = None):
        super().__init__(api_key=api_key)
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)

    def pre_process(self, model: ModelCard, messages: List[Message], tools: List[ToolDesc], stream: bool, **kwargs) -> Tuple[List[dict], dict]:
        messages, tools, kwargs = super().pre_process(model=model, messages=messages, tools=tools, stream=stream, **kwargs)
        for message in messages:
            # logger.debug(f"{message=}")
            if message.get("image"):
                base64 = image2base64(message["image"])
                message["content"] = [dict(type="text", text=message["content"]),
                                      dict(type="image_url", image_url=dict(url="data:image/jpeg;base64," + base64))]
                del message["image"]
        return messages, tools, kwargs

    def _inner_complete_(self, model: str, messages: List[dict], stream: bool, tools: List[dict], ** kwargs) -> Any:
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream,
            **kwargs
        )
        logger.debug(f"{response=}")
        return response

    def post_process(self, response, **kwargs) -> ModelResponse:
        content = response.choices[0].message.content
        usage = Usage(**response.usage.model_dump())
        return ModelResponse(content=content, usage=usage)

    def post_process_stream(self, response, **kwargs) -> ModelResponse:
        gen = (e for e in (get_text_chunk(chunk) for chunk in response) if e)
        gen = add_callback2gen(gen, acc_chunks)
        return ModelResponse(content=gen)


if __name__ == "__main__":
    provider = OpenAIProvider()
    messages = [Message(role="user", content="你好")]
    resp = provider.complete(messages=messages, model="gpt-4o-mini", stream=False)
    print(resp.content)
