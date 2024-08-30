#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/06/25 18:52:00
@Author  :   ChenHao
@Description  :   
@Contact :   jerrychen1990@gmail.com
'''


from typing import Any, Dict, List, Tuple

from litellm import completion
from loguru import logger

from liteai.core import ModelCard, ModelResponse, Message, ToolDesc, Usage
from liteai.provider.base import BaseProvider
from snippets.utils import add_callback2gen


from litellm import completion

from liteai.utils import acc_chunks, get_text_chunk


# class ZhipuTGILLM(CustomLLM):
#     pass


# zhipu_tgi_llm = ZhipuTGILLM()


# litellm.custom_provider_map = [  # 👈 KEY STEP - REGISTER HANDLER
#     {"provider": "zhipu_tgi", "custom_handler": zhipu_tgi_llm}
# ]


class TGIProvider(BaseProvider):
    key = "tgi"
    # TODO 更合理的控制参数
    allow_kwargs = {"stream", "temperature", "top_p", "max_tokens", "message_show_type"}

    def __init__(self, base_url: str = None, **kwargs):
        self.base_url = base_url

    def _support_system(self, model: str):
        return True

    def _parse_zhipu_message(self, messages: List[Message]) -> List[Dict]:
        _input = ""
        for message in messages:
            # logger.debug(f"{message=}")
            _input += f"<|{message['role']}|>{message['content']}"
        _input += "<|assistant|>"
        messages = [dict(content=_input, role="user")]
        return messages

    def pre_process(self, model: ModelCard, messages: List[Message], tools: List[ToolDesc], stream: bool, **kwargs) -> Tuple[List[dict], dict]:
        messages, tools, kwargs = super().pre_process(model=model, messages=messages, tools=tools, stream=stream, **kwargs)
        model_name = model.name.lower()
        if "zhipu" in model_name or "glm" in model_name:
            logger.debug("parse tgi input for zhipu model")
            messages = self._parse_zhipu_message(messages)
            kwargs.update({"stop":  ["<|endoftext|>", "<|user|>", "<|observation|>"]})
        return messages, tools, kwargs

    def _parse_model(self, model: str) -> str:
        return "huggingface/" + model

    def _inner_complete_(self, model: str, messages: List[dict], stream: bool, tools, ** kwargs) -> Any:
        model = self._parse_model(model)
        api_base = self.base_url + "/generate" if not stream else self.base_url + "/generate_stream"
        response = completion(
            model=model,
            messages=messages,
            api_base=api_base,
            stream=stream,
            **kwargs
        )
        return response

    def post_process(self, response, **kwargs) -> ModelResponse:
        # logger.debug(f"{response=}")
        content = response.choices[0].message.content
        stop = kwargs.get("stop", [])
        for s in stop:
            if s in content:
                content = content[:content.index(s)]
        usage = Usage(**response.usage.model_dump())

        return ModelResponse(content=content, usage=usage)

    def post_process_stream(self, response, **kwargs) -> ModelResponse:
        gen = (e for e in (get_text_chunk(chunk) for chunk in response) if e)
        stops = kwargs.get("stop", [])
        gen = (e for e in gen if e and not e in stops)
        gen = add_callback2gen(gen, acc_chunks)
        return ModelResponse(content=gen)
