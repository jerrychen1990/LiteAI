import unittest

from unillm.core import Message
from unillm.api import chat
from unillm.utils import set_logger, show_response
from loguru import logger


class TestZhipu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_logger(__name__)
        logger.info("start test job")

    def test_sync(self):
        system = "用英文回答我的问题, 80个单词以内"
        question = "介绍三首歌"
        model = "glm-4-air"
        messages = [Message(role="system", content=system),
                    Message(role="user", content=question)]
        response = chat(model=model, messages=messages, stream=False, temperature=0.)
        show_response(response)
        self.assertIsNotNone(response.usage)
        messages.extend([Message(role="assistant", content=response.content),
                        Message(role="user", content="介绍第二首歌")])
        response = chat(model=model, messages=messages, stream=False, temperature=0.)
        show_response(response)
        self.assertIsNotNone(response.usage)
        self.assertTrue("Hotel California" in response.content)

    def test_stream(self):
        question = "作一首五言绝句"
        model = "glm-3-turbo"
        messages = [Message(role="user", content=question)]
        resp = chat(model=model, messages=messages, stream=True, temperature=0.6)
        show_response(resp)