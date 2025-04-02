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


class TestDeepSeek(BasicTestCase):
    @classmethod
    def setUpClass(cls):
        set_logger(__name__)
        logger.info("start test deepseek")

    def test_basic_llm(self):
        super().basic_llm("deepseek-chat")
