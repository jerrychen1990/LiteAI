#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/09/03 15:14:06
@Author  :   ChenHao
@Description  :   对接ollama模型
@Contact :   jerrychen1990@gmail.com
'''


from liteai.utils import set_logger
from loguru import logger

from tests.base import BasicTestCase


class TestSiliconflow(BasicTestCase):

    @classmethod
    def setUpClass(cls):
        set_logger(__name__)
        logger.info("start test siliconflow job")

    def test_basic_llm(self):
        super().basic_llm(model="deepseek-ai/DeepSeek-V2-Chat")
