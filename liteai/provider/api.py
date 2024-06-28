#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/06/25 16:53:45
@Author  :   ChenHao
@Description  :   
@Contact :   jerrychen1990@gmail.com
'''

from typing import List
from liteai.core import BaseProvider
from liteai.provider.zhipu import ZhipuProvider
from liteai.provider.qwen import QwenProvider
from loguru import logger

_ALL_PROVIDERS: List[BaseProvider] = [ZhipuProvider, QwenProvider]
_PROVIDER_MAP = {p.key: p for p in _ALL_PROVIDERS}


def get_provider(provider_name: str, model_name: str, **kwargs) -> BaseProvider:
    if not provider_name:
        logger.debug("No provider specified, inferring from model name")
        model_name = model_name.lower()
        if "glm" in model_name:
            provider_name = ZhipuProvider.key
        if "qwen" in model_name:
            provider_name = QwenProvider.key

    if provider_name not in _PROVIDER_MAP:
        raise ValueError(f"Unsupported provider: {provider_name}")

    provider = _PROVIDER_MAP[provider_name]
    return provider(**kwargs)
