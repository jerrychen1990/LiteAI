#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/06/25 15:54:25
@Author  :   ChenHao
@Description  :   
@Contact :   jerrychen1990@gmail.com
'''


from .core import *
from .provider.api import get_provider
from typing import List


def chat(model: str, messages: List[dict | Message],
         provider: str = None, api_key=None, stream=False,
         temperature=0.7, top_p=.07,
         **kwargs) -> ModelResponse:
    provider: BaseProvider = get_provider(provider_name=provider, model_name=model, api_key=api_key)
    response = provider.complete(messages=messages, model=model, stream=stream, **kwargs)
    return response
