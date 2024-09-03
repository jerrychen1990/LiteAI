#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/08/21 18:22:06
@Author  :   ChenHao
@Description  :   
@Contact :   jerrychen1990@gmail.com
'''


from loguru import logger
from liteai.provider.api import get_provider_key
from liteai.core import ModelCard, ModelType

ZHIPU_MODELS = [
    ModelCard(name="glm-4-0520", description="glm-4-0520", provider="zhipu"),
    ModelCard(name="glm-4-air", description="glm-4-air", provider="zhipu"),
    ModelCard(name="glm-4-airx", description="glm-4-airx", provider="zhipu"),
    ModelCard(name="glm-4-flash", description="glm-4-flash", provider="zhipu"),
    ModelCard(name="glm-4v", description="glm-4v", provider="zhipu", support_vision=True),
    ModelCard(name="glm-4-plus", description="glm-4-plus", provider="zhipu", support_vision=False),
    ModelCard(name="glm-4-long", description="glm-4-long", provider="zhipu", support_vision=False),

    ModelCard(name="emohaa", description="emohaa", provider="zhipu", support_vision=True),

    ModelCard(name="embedding-2", description="embedding-2", provider="zhipu", model_type=ModelType.EMBEDDING),
    ModelCard(name="embedding-3", description="embedding-2", provider="zhipu", model_type=ModelType.EMBEDDING),
]

OPENAI_MODELS = [
    ModelCard(name="gpt-4o", description="gpt-4o", provider="openai"),
    ModelCard(name="gpt-4", description="gpt-4", provider="openai"),
    ModelCard(name="gpt-4o-mini", description="gpt-4o-mini", provider="openai")
]


MINIMAX_MODELS = [
    ModelCard(name="abab6.5s-chat", description="abab6.5s-chat", provider="minimax"),
    ModelCard(name="speech-01-turbo", description="speech-01-turbo", provider="minimax", model_type=ModelType.TTS)
]

DOUBAO_MODELS = [
    ModelCard(name="doubao-lite-4k", description="doubao-lite-4k", provider="doubao"),
]

QWEN_MODELS = [
    ModelCard(name="qwen-turbo", description="qwen-turbo", provider="qwen"),
    ModelCard(name="qwen-vl-plus", description="qwen-vl-plus", provider="qwen", support_vision=True)
]


ALL_MODELS = ZHIPU_MODELS + OPENAI_MODELS + MINIMAX_MODELS + QWEN_MODELS + DOUBAO_MODELS
ALL_MODEL_MAP = {model.name: model for model in ALL_MODELS}


def get_modelcard(model_name: str, **kwargs) -> ModelCard:
    model_name = model_name.lower()
    if model_name not in ALL_MODEL_MAP:
        logger.debug(f"get not defined model name:{model_name}")
        provider_key = kwargs.pop("provider_key", None)
        if not provider_key:
            provider_key = get_provider_key(model_name=model_name)
        logger.debug(f"provider_key: {provider_key}")

        model_card = ModelCard(name=model_name, description=model_name, provider=provider_key)
        return model_card
    return ALL_MODEL_MAP[model_name]


# __all__ = ALL_MODELS + [ALL_MODEL_MAP, get_modelcard, ALL_MODELS]
