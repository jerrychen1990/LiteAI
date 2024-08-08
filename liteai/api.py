#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/06/25 15:54:25
@Author  :   ChenHao
@Description  :
@Contact :   jerrychen1990@gmail.com
'''


from liteai.config import ALL_MODELS
from liteai.core import *
from liteai.provider.api import get_provider
from typing import List

from liteai.provider.base import BaseProvider
from snippets import ChangeLogLevelContext


def chat(model: str, messages: List[dict | Message] | str,
         provider: str = None, api_key: str = None, base_url: str = None,
         stream=False, temperature=0.7, top_p=.07, log_level: str = None,
         **kwargs) -> ModelResponse:

    def _chat():
        _provider: BaseProvider = get_provider(provider_name=provider, model_name=model, api_key=api_key, base_url=base_url)
        if isinstance(messages, str):
            _messages = [Message(content=messages, role="user")]
        else:
            _messages = [Message(**m) if isinstance(m, dict) else m for m in messages]

        response = _provider.complete(messages=_messages, model=model, stream=stream,
                                      temperature=temperature, top_p=top_p, **kwargs)
        return response

    if log_level:
        with ChangeLogLevelContext(module_name="liteai", sink_type="stdout", level=log_level.upper()):
            response = _chat()
    else:
        response = _chat()
    return response


def tts(text: str | Iterable[str], model: str,
        provider: str = None, api_key: str = None,
        tgt_path: str = None, stream=True, **kwargs) -> Voice:

    provider: BaseProvider = get_provider(provider_name=provider, model_name=model, api_key=api_key)
    voice = provider.tts(text=text, model=model, stream=stream, tgt_path=tgt_path, **kwargs)
    return voice


def list_models() -> List[str]:
    return ALL_MODELS
