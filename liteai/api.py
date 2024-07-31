#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/06/25 15:54:25
@Author  :   ChenHao
@Description  :   
@Contact :   jerrychen1990@gmail.com
'''


from liteai.core import *
from liteai.provider.api import get_provider
from typing import List

from liteai.provider.base import BaseProvider


def chat(model: str, messages: List[dict | Message],
         provider: str = None, api_key: str = None, base_url: str = None,
         stream=False, temperature=0.7, top_p=.07,
         **kwargs) -> ModelResponse:
    provider: BaseProvider = get_provider(provider_name=provider, model_name=model, api_key=api_key, base_url=base_url)
    assert messages
    messages = [Message(**m) if isinstance(m, dict) else m for m in messages]

    response = provider.complete(messages=messages, model=model, stream=stream,
                                 temperature=temperature, top_p=top_p, **kwargs)
    return response
