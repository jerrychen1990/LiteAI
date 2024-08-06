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


from liteai.core import ModelResponse, Message, Usage
from liteai.provider.base import BaseProvider
from openai import OpenAI
from snippets.utils import add_callback2gen

from liteai.utils import image2base64

from snippets import add_callback2gen

from liteai.utils import get_chunk_data, image2base64, acc_chunks


class OpenAIProvider(BaseProvider):
    key: str = "openai"
    allow_kwargs = {"do_sample", "stream", "temperature", "top_p", "max_tokens"}
    api_key_env = "OPENAI_API_KEY"

    def __init__(self, api_key: str = None, base_url: str = None):
        super().__init__(api_key=api_key)
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)

    def _support_system(self, model: str):
        return True

    def pre_process(self, model: str, messages: List[Message], stream: bool, **kwargs) -> Tuple[List[dict], dict]:
        messages, kwargs = super().pre_process(model, messages, stream, **kwargs)
        for message in messages:
            # logger.debug(f"{message=}")
            if message.get("image"):
                base64 = image2base64(message["image"])
                message["content"] = [dict(type="text", text=message["content"]),
                                      dict(type="image_url", image_url=dict(url="data:image/jpeg;base64," + base64))]
                del message["image"]
        return messages, kwargs

    def _inner_complete_(self, model, messages: List[dict], stream: bool, ** kwargs) -> Any:
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream,
            **kwargs
        )
        logger.debug(f"{response=}")
        return response

    def post_process(self, response) -> ModelResponse:
        content = response.choices[0].message.content
        usage = Usage(**response.usage.model_dump())
        return ModelResponse(content=content, usage=usage)

    def post_process_stream(self, response) -> ModelResponse:
        gen = (e for e in (get_chunk_data(chunk) for chunk in response) if e)
        gen = add_callback2gen(gen, acc_chunks)
        return ModelResponse(content=gen)


if __name__ == "__main__":
    provider = OpenAIProvider()
    messages = [Message(role="user", content="你好")]
    resp = provider.complete(messages=messages, model="gpt-4o-mini", stream=False)
    print(resp.content)
