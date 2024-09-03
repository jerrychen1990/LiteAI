#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/07/31 15:22:33
@Author  :   ChenHao
@Description  : OpenAI测试
@Contact :   jerrychen1990@gmail.com
'''

import os

from liteai.core import Voice
from liteai.api import tts
from liteai.utils import set_logger
from liteai.voice import play_voice
from loguru import logger

from tests.base import BasicTestCase


class TestMiniMax(BasicTestCase):
    @classmethod
    def setUpClass(cls):
        set_logger(__name__)
        logger.info("start test minimax")

    def test_basic_llm(self):
        super().basic_llm(model="abab6.5s-chat")

    def test_tts(self):
        text = "你好呀，我是liteai"
        tgt_path = os.path.abspath(os.path.dirname(__file__)) + "/tmp/test_minimax_1.mp3"
        voice: Voice = tts(text=text, model="speech-01-turbo", tgt_path=tgt_path, stream=False)
        play_voice(voice)
        os.remove(tgt_path)
        text = "你好呀，我是流式的一段话"
        tgt_path = os.path.abspath(os.path.dirname(__file__)) + "/tmp/test_minimax_2.mp3"
        voice: Voice = tts(text=text, model="speech-01-turbo", tgt_path=tgt_path, stream=True)
        play_voice(voice)
        play_voice(voice)
        os.remove(tgt_path)
