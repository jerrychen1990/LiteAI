#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/07/31 15:22:33
@Author  :   ChenHao
@Description  : OpenAI测试
@Contact :   jerrychen1990@gmail.com
'''

import os
import unittest

from liteai.core import Message, Voice
from liteai.api import chat, tts
from liteai.utils import set_logger, show_response
from liteai.voice import play_voice
from loguru import logger


class TestMiniMax(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_logger(__name__)
        logger.info("start test minimax")

    def test_sync(self):
        system = "用英文回答我的问题, 80个单词以内"
        question = "列出国土面积最大的五个国家"
        model = "abab6.5s-chat"
        messages = [Message(role="system", content=system),
                    Message(role="user", content=question)]
        response = chat(model=model, messages=messages, stream=False, temperature=0.)
        show_response(response)
        self.assertIsNotNone(response.usage)
        messages.extend([Message(role="assistant", content=response.content),
                        Message(role="user", content="介绍第二个")])
        response = chat(model=model, messages=messages, stream=False, temperature=0.)
        show_response(response)
        self.assertIsNotNone(response.usage)
        self.assertTrue("Canada" in response.content)

    def test_stream(self):
        question = "作一首五言绝句"
        model = "abab6.5s-chat"
        messages = [Message(role="user", content=question)]
        resp = chat(model=model, messages=messages, stream=True, temperature=0.6)
        show_response(resp)

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
