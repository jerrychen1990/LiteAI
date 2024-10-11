#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Time    :   2024/07/31 15:22:33
@Author  :   ChenHao
@Description  : OpenAI测试
@Contact :   jerrychen1990@gmail.com
"""

import unittest

from loguru import logger

from liteai.api import chat
from liteai.core import Message
from liteai.utils import set_logger, show_response
from tests.base import BasicTestCase


class TestOpenAI(BasicTestCase):
    @classmethod
    def setUpClass(cls):
        set_logger(__name__)
        logger.info("start test openai")

    def test_basic_llm(self):
        super().basic_llm("gpt-3.5-turbo")

    def test_vision_chat(self):
        question = "这张图里有什么?"
        image_path = "./data/Pikachu.png"
        model = "gpt-4o-mini"
        messages = [Message(role="user", content=question, image=image_path)]
        resp = chat(model=model, messages=messages, stream=False, temperature=0.0)
        show_response(resp)
        self.assertTrue("皮卡丘" in resp.content)

    @unittest.skip("本地模型不一定部署")
    def test_local_model(self):
        # os.environ.pop("OPENAI_API_KEY")
        question = "作一首五言绝句"
        model = "glm4-9b-local"
        base_url = "http://36.103.167.117:8100/v1/"
        messages = [Message(role="user", content=question)]
        resp = chat(
            model=model, messages=messages, provider="openai", stream=False, temperature=0.6, base_url=base_url, max_tokens=128, api_key=""
        )
        show_response(resp)
