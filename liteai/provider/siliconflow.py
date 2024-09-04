#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/09/04 14:21:52
@Author  :   ChenHao
@Description  :   硅基流动
@Contact :   jerrychen1990@gmail.com
'''


import json
from dotenv import load_dotenv
from loguru import logger
import requests
from typing import Any, List


from liteai.core import ModelCard, ModelResponse, Message, Usage
from liteai.provider.base import BaseProvider
from snippets.logs import set_logger
from snippets.utils import add_callback2gen


from snippets import add_callback2gen

from liteai.utils import acc_chunks, show_response


class SiliconFlowProvider(BaseProvider):
    key: str = "siliconflow"
    allow_kwargs = {"stream", "temperature", "top_p", "max_tokens"}
    api_key_env = "SILICONFLOW_API_KEY"

    def __init__(self, api_key: str = None, base_url: str = None):
        super().__init__(api_key=api_key)

    def _inner_complete_(self, model: str, messages: List[dict], stream: bool, tools: List[dict], ** kwargs) -> Any:
        url = "https://api.siliconflow.cn/v1/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self.api_key}"
        }
        response = requests.post(url, json=payload, headers=headers, stream=stream)
        response.raise_for_status()
        if stream:
            lines = (e.decode("utf8") for e in response.iter_lines())
            lines = (e[len("data:"):] for e in lines if e)
            # lines = list(lines)
            # for line in lines:
            #     logger.debug(f"{line=}")
            return (json.loads(e) for e in lines if e != " [DONE]")
        else:
            return response.json()

    def post_process(self, response: dict, **kwargs) -> ModelResponse:
        content = response["choices"][0]["message"]["content"]
        usage = Usage(**response["usage"])
        return ModelResponse(content=content, usage=usage)

    def post_process_stream(self, response, **kwargs) -> ModelResponse:
        def _get_text_chunk(chunk: dict) -> str:
            delta = chunk["choices"][0]["delta"]
            return delta.get("content")

        gen = (e for e in (_get_text_chunk(chunk) for chunk in response) if e)
        gen = add_callback2gen(gen, acc_chunks)
        return ModelResponse(content=gen)


if __name__ == "__main__":
    load_dotenv()
    set_logger("DEV", __name__)
    provider = SiliconFlowProvider()
    messages = [Message(role="user", content="你好")]
    model = ModelCard(name="deepseek-ai/DeepSeek-V2-Chat")
    resp = provider.complete(messages=messages, model=model, stream=True)
    show_response(resp)
