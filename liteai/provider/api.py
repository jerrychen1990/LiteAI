#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/06/25 16:53:45
@Author  :   ChenHao
@Description  :
@Contact :   jerrychen1990@gmail.com
'''

from typing import List
from liteai.core import ModelCard
from liteai.provider.doubao import DoubaoProvider
from liteai.provider.minimax import MinimaxProvider
from liteai.provider.tgi import TGIProvider
from liteai.provider.zhipu import ZhipuProvider
from liteai.provider.qwen import QwenProvider
from liteai.provider.open_ai import OpenAIProvider
from liteai.provider.lite_llm import LiteLLMProvider
from liteai.provider.xunfei import XunfeiProvider
from liteai.provider.siliconflow import SiliconFlowProvider


from liteai.provider.base import BaseProvider
from dotenv import load_dotenv
load_dotenv()  # 默认会加载当前目录下的 .env 文件


_ALL_PROVIDERS: List[BaseProvider] = [ZhipuProvider, QwenProvider, OpenAIProvider,
                                      DoubaoProvider, MinimaxProvider, LiteLLMProvider, TGIProvider, XunfeiProvider, SiliconFlowProvider]
_PROVIDER_MAP = {p.key: p for p in _ALL_PROVIDERS}


def get_provider_key(model_name: str) -> str:
    model_name = model_name.lower()
    if "tgi" in model_name:
        provider_name = TGIProvider.key
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
    elif "xunfei" in model_name:
        provider_name = XunfeiProvider.key
    elif "ollama" in model_name:
        provider_name = LiteLLMProvider.key
    elif "deepseek" in model_name:
        provider_name = SiliconFlowProvider.key
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return provider_name


def get_provider(model: ModelCard, api_key: str = None, base_url: str = None, **kwargs):
    provider_name = model.provider
    if provider_name not in _PROVIDER_MAP:
        raise ValueError(f"Unsupported provider: {provider_name}")

    provider_cls = _PROVIDER_MAP[provider_name]
    provider = provider_cls(api_key=api_key, base_url=base_url, **kwargs)
    return provider
