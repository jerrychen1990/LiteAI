import unittest

from liteai.core import Message
from liteai.api import chat
from liteai.utils import set_logger, show_response
from loguru import logger


class TestZhipu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_logger(__name__)
        logger.info("start test job")

    def test_sync(self):
        system = "用英文回答我的问题, 80个单词以内"
        question = "列出国土面积最大的五个国家"
        model = "glm-4-air"
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
        model = "glm-3-turbo"
        messages = [Message(role="user", content=question)]
        resp = chat(model=model, messages=messages, stream=True, temperature=0.6)
        show_response(resp)

    def test_vision_chat(self):
        question = "这张图里有什么?"
        image_path = "./data/Pikachu.png"
        model = "glm-4v"
        messages = [Message(role="user", content=question, image=image_path)]
        resp = chat(model=model, messages=messages, stream=False, temperature=0.)
        show_response(resp)
        self.assertTrue("皮卡丘" in resp.content)

    def test_not_support_system_model(self):
        system = "用英文回答我的问题, 80个单词以内"
        question = "列出国土面积最大的五个国家"
        model = "chatglm_12b"
        messages = [Message(role="system", content=system),
                    Message(role="user", content=question)]
        response = chat(model=model, messages=messages, stream=False, temperature=0.)
        show_response(response)

    def test_lingxin_model(self):
        model = "emohaa"
        meta = {
            "user_info": "30岁的男性软件工程师，兴趣包括阅读、徒步和编程",
            "bot_info": "Emohaa是一款基于Hill助人理论的情感支持AI，拥有专业的心理咨询话术能力",
            "bot_name": "Emohaa",
            "user_name": "张三"
        }
        messages = [
            {"role": "system", "content": "你的名字叫Aifori"},
            {
                "role": "assistant",
                "content": "你好，我是Emohaa，很高兴见到你。请问有什么我可以帮忙的吗？"
            },
            {
                "role": "user",
                "content": "最近我感觉压力很大，情绪总是很低落。"
            },
            {
                "role": "assistant",
                "content": "听起来你最近遇到了不少挑战。可以具体说说是什么让你感到压力大吗？"
            },
            {
                "role": "user",
                "content": "主要是工作上的压力，任务太多，总感觉做不完。"
            }
        ]
        resp = chat(model=model, messages=messages, stream=False, meta=meta)
        logger.info(f"{resp.content=}")
