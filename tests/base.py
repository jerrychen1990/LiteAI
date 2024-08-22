#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/08/22 10:54:26
@Author  :   ChenHao
@Description  :   
@Contact :   jerrychen1990@gmail.com
'''


import unittest

from liteai.api import chat
from liteai.core import Message
from liteai.utils import show_response


class BasicTestCase(unittest.TestCase):

    @unittest.skip("基础函数，不测试")
    def test_basic_llm(self, model):
        system = "用英文回答我的问题, 80个单词以内"
        question = "列出国土面积最大的五个国家"
        messages = [Message(role="system", content=system),
                    Message(role="user", content=question)]
        response = chat(model=model, messages=messages, stream=False, temperature=0.)
        show_response(response)
        self.assertIsNotNone(response.usage)
        messages.extend([Message(role="assistant", content=response.content),
                        Message(role="user", content="介绍第二个")])
        response = chat(model=model, messages=messages, stream=True, temperature=0.)
        content = show_response(response)
        # self.assertIsNotNone(response.usage)
        self.assertIn("Canada", content)
