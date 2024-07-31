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

from liteai.utils import image2base64


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
        def _gen():
            acc = []
            for chunk in response:
                # logger.debug(f"{chunk=}")
                choices = chunk.choices
                if choices:
                    choice = choices[0]
                    if choice.delta.content:
                        delta_content = choice.delta.content
                        # logger.info(f"{delta_content}")
                        yield delta_content
                        acc.append(delta_content)
                _finish_reason = choice.finish_reason
                if _finish_reason:
                    if _finish_reason == "sensitive":
                        logger.warning(f"zhipu api finish with reason {_finish_reason}")
                        msg = "系统检测到输入或生成内容可能包含不安全或敏感内容，请您避免输入易产生敏感内容的提示语，感谢您的配合。"
                        acc.append(msg)
                        yield msg

            resp_msg = "".join(acc).strip()
            logger.debug(f"model generate answer:{resp_msg}")
        return ModelResponse(content=_gen())


if __name__ == "__main__":
    provider = OpenAIProvider()
    messages = [Message(role="user", content="你好")]
    resp = provider.complete(messages=messages, model="gpt-4o-mini", stream=False)
    print(resp.content)
