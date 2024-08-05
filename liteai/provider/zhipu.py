#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/06/25 16:39:05
@Author  :   ChenHao
@Description  :   
@Contact :   jerrychen1990@gmail.com
'''
from typing import Any, List, Tuple


from liteai.core import ModelResponse, Message, Usage
from zhipuai import ZhipuAI
from liteai.provider.base import BaseProvider
from snippets import add_callback2gen

from liteai.utils import get_chunk_data, image2base64, acc_chunks


class ZhipuProvider(BaseProvider):
    key: str = "zhipu"
    allow_kwargs = {"do_sample", "stream", "temperature", "top_p", "max_tokens", "meta"}
    api_key_env = "ZHIPUAI_API_KEY"

    def __init__(self, api_key: str = None, **kwargs):
        super().__init__(api_key=api_key)
        self.client = ZhipuAI(api_key=self.api_key)

    def _support_system(self, model: str):
        model = model.lower()
        if "glm-4" in model:
            return True
        return "chatglm3" in model or "glm-3" in model

    def pre_process(self, model: str, messages: List[Message], stream: bool, **kwargs) -> Tuple[List[dict], dict]:
        if kwargs.get("temperature") == 0.:
            del kwargs["temperature"]
            kwargs["do_sample"] = False
        messages, kwargs = super().pre_process(model, messages, stream, **kwargs)
        for message in messages:
            # logger.debug(f"{message=}")
            if message.get("image"):
                base64 = image2base64(message["image"])
                message["content"] = [dict(type="text", text=message["content"]),
                                      dict(type="image_url", image_url=dict(url=base64))]
                del message["image"]
        return messages, kwargs

    def _inner_complete_(self, model, messages: List[dict], stream: bool, ** kwargs) -> Any:
        # logger.debug(f"{self.client.api_key=}")
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream,
            **kwargs
        )
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
    provider = ZhipuProvider()
    messages = [Message(role="user", content="你好")]
    resp = provider.complete(messages=messages, model="glm-3-turbo", stream=False)
    print(resp.content)
