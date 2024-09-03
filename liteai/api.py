#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/06/25 15:54:25
@Author  :   ChenHao
@Description  :
@Contact :   jerrychen1990@gmail.com
'''


from functools import wraps

from loguru import logger


from liteai.core import *
from liteai.modelcard import ALL_MODELS, get_modelcard
from liteai.provider.api import get_provider
from typing import List

from liteai.provider.base import BaseProvider
from liteai.tool import on_tool_call
from snippets import ChangeLogLevelContext


def can_set_level(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        log_level = kwargs.pop("log_level", None)
        if log_level:
            with ChangeLogLevelContext(module_name="liteai", sink_type="stdout", level=log_level.upper()):
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return wrapper


@can_set_level
def chat(model: str | ModelCard, messages: List[dict | Message] | str, tool_calls: List[ToolCall] = [],
         provider: str = None, api_key: str = None, base_url: str = None, tools: List[ToolDesc] = [],
         stream=False, temperature=0.7, top_p=.7,  **kwargs) -> ModelResponse:
    if isinstance(model, str):
        model: ModelCard = get_modelcard(model_name=model, provider_key=provider)

    provider: BaseProvider = get_provider(model=model, api_key=api_key, base_url=base_url)

    if isinstance(messages, str):
        messages = [Message(content=messages, role="user")]
    else:
        messages = [Message(**m) if isinstance(m, dict) else m for m in messages]

    response = provider.complete(messages=messages, model=model, stream=stream, tools=tools, tool_calls=tool_calls,
                                 temperature=temperature, top_p=top_p, **kwargs)
    return response


@can_set_level
def tts(text: str | Iterable[str], model: str | ModelCard,
        provider: str = None, api_key: str = None,
        tgt_path: str = None, stream=True, **kwargs) -> Voice:
    if isinstance(model, str):
        model: ModelCard = get_modelcard(model_name=model, provider_key=provider)
    provider: BaseProvider = get_provider(model=model, api_key=api_key)
    voice = provider.tts(text=text, model=model, stream=stream, tgt_path=tgt_path, **kwargs)
    return voice


@can_set_level
def asr(voice: Voice, model: str | ModelCard, provider: str = None, api_key: str = None,
        **kwargs) -> str:
    if isinstance(model, str):
        model: ModelCard = get_modelcard(model_name=model, provider_key=provider)
    provider: BaseProvider = get_provider(model=model, api_key=api_key)
    voice = provider.asr(voice=voice, model=model, **kwargs)
    return voice


@can_set_level
def embedding(texts: str | List[str], model: str | ModelCard,
              provider: str = None, api_key: str = None, norm=True, batch_size=8, **kwargs) -> List[List[float]] | List[float]:
    if isinstance(model, str):
        model: ModelCard = get_modelcard(model_name=model, provider_key=provider)
    provider: BaseProvider = get_provider(model=model, api_key=api_key)
    voice = provider.embedding(texts=texts, model=model, norm=norm, batch_size=batch_size, **kwargs)
    return voice


@can_set_level
def agent_chat(model: str | ModelCard, messages: List[dict | Message] | str, provider: str = None,
               api_key: str = None, base_url: str = None, tools: List[ToolDesc] = [], max_iter=3,
               stream=False, temperature=0.7, top_p=.7,  **kwargs) -> ModelResponse:
    tool_calls = []

    idx = 0
    while idx < max_iter:
        logger.info(f"agent chat iteration:{idx+1}")
        resp = chat(model=model, messages=messages, provider=provider, api_key=api_key, base_url=base_url, tools=tools, tool_calls=tool_calls,
                    stream=stream, temperature=temperature, top_p=top_p, **kwargs)
        tool_calls = resp.tool_calls
        for tool_call in tool_calls:
            on_tool_call(tool_call)
        tool_calls = [e for e in tool_calls if e.tool_desc.is_inner and e.resp]
        logger.info(f"get {len(tool_calls)} inner tool calls")
        if not tool_calls:
            return resp
        idx += 1
    return resp


def list_models() -> List[dict]:
    return [model.model_dump() for model in ALL_MODELS]
