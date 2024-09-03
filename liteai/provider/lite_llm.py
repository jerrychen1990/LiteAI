#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/06/25 18:52:00
@Author  :   ChenHao
@Description  :   
@Contact :   jerrychen1990@gmail.com
'''


from typing import Any, List

from litellm import completion
from loguru import logger

from liteai.core import ModelResponse, Message, Usage
from liteai.provider.base import BaseProvider
from snippets.utils import add_callback2gen

from liteai.utils import acc_chunks, get_text_chunk


class LiteLLMProvider(BaseProvider):
    key: str = "litellm"
    allow_kwargs = {"stream", "temperature", "top_p", "max_tokens"}
    api_key_env = "LLM_API_KEY"

    def __init__(self, api_key: str = None, base_url: str = None):
        self.base_url = base_url
        super().__init__(api_key=api_key)

    # def pre_process(self, model: str, messages: List[Message], tools: List[ToolDesc], stream: bool, **kwargs) -> Tuple[List[dict], dict]:
    #     messages, tools, kwargs = super().pre_process(model=model, messages=messages, tools=tools, stream=stream, **kwargs)
    #     for message in messages:
    #         # logger.debug(f"{message=}")
    #         if message.get("image"):
    #             base64 = image2base64(message["image"])
    #             message["content"] = [dict(type="text", text=message["content"]),
    #                                   dict(type="image_url", image_url=dict(url="data:image/jpeg;base64," + base64))]
    #             del message["image"]
    #     return messages, tools, kwargs

    def _inner_complete_(self, model: str, messages: List[dict], stream: bool, tools, ** kwargs) -> Any:
        # model = self._parse_model(model)
        # logger.debug(f"{model=}")
        response = completion(
            model=model,
            messages=messages,
            api_base=self.base_url,
            stream=stream,
            **kwargs
        )
        return response

    def post_process(self, response, **kwargs) -> ModelResponse:
        logger.debug(f"{response=}")
        content = response.choices[0].message.content
        usage = Usage(**response.usage.model_dump())
        return ModelResponse(content=content, usage=usage)

    def post_process_stream(self, response, **kwargs) -> ModelResponse:
        gen = (e for e in (get_text_chunk(chunk) for chunk in response) if e)
        gen = add_callback2gen(gen, acc_chunks)
        return ModelResponse(content=gen)


if __name__ == "__main__":
    provider = LiteLLMProvider(base_url="http://36.103.167.117:8101")
    messages = [Message(role="user", content="你好")]
    resp = provider.complete(messages=messages, model="tgi_glm2_12b", stream=False)
    print(resp.content)
