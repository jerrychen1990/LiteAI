#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/06/25 16:53:45
@Author  :   ChenHao
@Description  :   
@Contact :   jerrychen1990@gmail.com
'''

from typing import List
from liteai.provider.doubao import DoubaoProvider
from liteai.provider.minimax import MinimaxProvider
from liteai.provider.zhipu import ZhipuProvider
from liteai.provider.qwen import QwenProvider
from liteai.provider.open_ai import OpenAIProvider
from liteai.provider.lite_llm import LiteLLMProvider


from liteai.provider.base import BaseProvider
from loguru import logger
from dotenv import load_dotenv
load_dotenv()  # 默认会加载当前目录下的 .env 文件


_ALL_PROVIDERS: List[BaseProvider] = [ZhipuProvider, QwenProvider, OpenAIProvider, DoubaoProvider, MinimaxProvider]
_PROVIDER_MAP = {p.key: p for p in _ALL_PROVIDERS}


def get_provider(provider_name: str, model_name: str, **kwargs) -> BaseProvider:
    if not provider_name:
        logger.debug("No provider specified, inferring from model name")
        model_name = model_name.lower()
        if "tgi" in model_name:
            provider_name = LiteLLMProvider.key
        elif "glm" in model_name:
            provider_name = ZhipuProvider.key
        elif model_name in ["emohaa"]:
            provider_name = ZhipuProvider.key
        elif "qwen" in model_name:
            provider_name = QwenProvider.key
        elif "gpt" in model_name:
            provider_name = OpenAIProvider.key
        elif "doubao" in model_name:
            provider_name = DoubaoProvider.key
        elif "abab" in model_name or "speech-01" in model_name:
            provider_name = MinimaxProvider.key

    if provider_name not in _PROVIDER_MAP:
        raise ValueError(f"Unsupported provider: {provider_name}")

    provider = _PROVIDER_MAP[provider_name]
    return provider(**kwargs)
