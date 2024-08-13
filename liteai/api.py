#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/06/25 15:54:25
@Author  :   ChenHao
@Description  :
@Contact :   jerrychen1990@gmail.com
'''


from functools import wraps

from liteai.config import ALL_MODELS
from liteai.core import *
from liteai.provider.api import get_provider
from typing import List

from liteai.provider.base import BaseProvider
from snippets import ChangeLogLevelContext


def can_set_level(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        log_level = kwargs.pop("log_level", None)
        # logger.debug(f"set log level: {log_level}")
        if log_level:
            with ChangeLogLevelContext(module_name="liteai", sink_type="stdout", level=log_level):
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return wrapper


@can_set_level
def chat(model: str, messages: List[dict | Message] | str,
         provider: str = None, api_key: str = None, base_url: str = None, tools: List[ToolDesc] = [],
         stream=False, temperature=0.7, top_p=.07,  **kwargs) -> ModelResponse:
    provider: BaseProvider = get_provider(provider_name=provider, model_name=model, api_key=api_key, base_url=base_url)
    if isinstance(messages, str):
        messages = [Message(content=messages, role="user")]
    else:
        messages = [Message(**m) if isinstance(m, dict) else m for m in messages]

    response = provider.complete(messages=messages, model=model, stream=stream, tools=tools,
                                 temperature=temperature, top_p=top_p, **kwargs)
    return response


@can_set_level
def tts(text: str | Iterable[str], model: str,
        provider: str = None, api_key: str = None,
        tgt_path: str = None, stream=True, **kwargs) -> Voice:
    provider: BaseProvider = get_provider(provider_name=provider, model_name=model, api_key=api_key)
    voice = provider.tts(text=text, model=model, stream=stream, tgt_path=tgt_path, **kwargs)
    return voice


@can_set_level
def embedding(texts: str | List[str], model: str,
              provider: str = None, api_key: str = None, norm=True, batch_size=8, **kwargs) -> List[List[float]] | List[float]:
    provider: BaseProvider = get_provider(provider_name=provider, model_name=model, api_key=api_key)
    voice = provider.embedding(texts=texts, model=model, norm=norm, batch_size=batch_size, **kwargs)
    return voice


def list_models() -> List[str]:
    return ALL_MODELS
