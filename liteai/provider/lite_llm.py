#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/06/25 18:52:00
@Author  :   ChenHao
@Description  :   
@Contact :   jerrychen1990@gmail.com
'''


from typing import Any, List, Tuple

from litellm import completion
from loguru import logger

from liteai.core import ModelResponse, Message, Usage
from liteai.provider.base import BaseProvider
from snippets.utils import add_callback2gen

from liteai.utils import acc_chunks, get_chunk_data, image2base64


class LiteLLMProvider(BaseProvider):
    key: str = "litellm"
    allow_kwargs = {"do_sample", "stream", "temperature", "top_p", "max_tokens"}
    api_key_env = "OPENAI_API_KEY"

    def __init__(self, api_key: str = None, base_url: str = None):
        super().__init__(api_key=api_key)
        self.base_url = base_url

    def _support_system(self, model: str):
        return True

    def _get_custom_provider(self, model: str):
        if "tgi" in model:
            return "huggingface"

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

    def _parse_model(self, model: str) -> str:
        if "tgi" in model:
            return "huggingface/" + model

    def _inner_complete_(self, model: str, messages: List[dict], stream: bool, ** kwargs) -> Any:
        model = self._parse_model(model)
        response = completion(
            model=model,
            messages=messages,
            api_base=self.base_url,
            stream=stream,
            **kwargs
        )
        return response

    def post_process(self, response) -> ModelResponse:
        logger.debug(f"{response=}")
        content = response.choices[0].message.content
        usage = Usage(**response.usage.model_dump())
        return ModelResponse(content=content, usage=usage)

    def post_process_stream(self, response) -> ModelResponse:
        gen = (e for e in (get_chunk_data(chunk) for chunk in response) if e)
        gen = add_callback2gen(gen, acc_chunks)
        return ModelResponse(content=gen)


if __name__ == "__main__":
    provider = LiteLLMProvider(base_url="http://36.103.167.117:8101")
    messages = [Message(role="user", content="你好")]
    resp = provider.complete(messages=messages, model="tgi_glm2_12b", stream=False)
    print(resp.content)
