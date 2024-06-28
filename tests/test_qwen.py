import unittest

from liteai.core import Message
from liteai.api import chat
from liteai.utils import set_logger, show_response
from loguru import logger


class TestQwen(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_logger(__name__)
        logger.info("start test job")

    def test_sync(self):
        system = "用英文回答我的问题, 80个单词以内"
        question = "列出国土面积最大的五个国家"
        model = "qwen-turbo"
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
        model = "qwen-turbo"
        messages = [Message(role="user", content=question)]
        resp = chat(model=model, messages=messages, stream=True, temperature=0.6)
        show_response(resp)

    def test_vision_chat(self):
        question = "这张图里有什么?"
        image_path = "./data/Pikachu.png"
        model = "qwen-vl-plus"
        messages = [Message(role="user", content=question, image=image_path)]
        resp = chat(model=model, messages=messages, stream=False, temperature=0.)
        show_response(resp)
        # self.assertTrue("皮卡丘" in resp.content)
