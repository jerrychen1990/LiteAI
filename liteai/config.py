#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/07/29 11:20:33
@Author  :   ChenHao
@Description  :   配置信息
@Contact :   jerrychen1990@gmail.com
'''


import os

LITEAI_ENV = os.environ.get("LITEAI_ENV", "dev")

LITEAI_HOME = os.environ.get("LITEAI_HOME", os.path.dirname(os.path.abspath(__file__)))
LOG_HOME = os.environ.get("LOG_HOME", os.path.join(LITEAI_HOME, "logs"))

ARK_ENDPOINT_MAP = {
    "doubao-lite-4k": "ep-20240805191433-4jh54"
}


ZHIPU_MODELS = ["glm-4-0520", "glm-4v", "glm-4-air", "glm-4-flash"]
OPENAI_MODELS = ["gpt-4o", "gpt-4", "gpt-4o-mini"]
MINIMAX_MODELS = ["abab6.5s-chat", "speech-01-turbo"]
QWEN_MODELS = ["qwen-turbo", "qwen-vl-plus"]

ALL_MODELS = ZHIPU_MODELS + OPENAI_MODELS + MINIMAX_MODELS + QWEN_MODELS

DEFAULT_VOICE_CHUNK_SIZE = 4096 * 10
MIN_PLAY_VOICE_SIZE = 8192 * 10
MAX_PLAY_SECONDS = 60 * 60
