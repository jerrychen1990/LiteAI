

import unittest
from liteai.core import Message
from liteai.api import agent_chat, chat, embedding
from liteai.tool import CurrentContextToolDesc, on_tool_call
from liteai.utils import get_embd_similarity, set_logger, show_embeds, show_response
from loguru import logger

from tests.base import BasicTestCase


class TestZhipu(BasicTestCase):
    @classmethod
    def setUpClass(cls):
        set_logger(__name__)
        logger.info("start test job")

    def test_basic_llm(self):
        super().basic_llm("glm-4-air")

    def test_vision_chat(self):
        super().basic_vision("glm-4v-plus")

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

    def test_agent_chat(self):
        question = "今天是几号了"
        model = "glm-4-plus"
        tools = [CurrentContextToolDesc]
        response = agent_chat(model=model, messages=question, tools=tools, stream=True, temperature=0., log_level="DEBUG")
        content = show_response(response)
        import datetime

        # 获取今天的日期
        today = datetime.date.today()
        # 以指定格式输出日期
        formatted_date = today.strftime("%-m月%-d日")
        logger.info(f"{formatted_date=}")
        self.assertTrue(formatted_date in content)

    @unittest.skip("skip test tool call")
    def test_tool_call(self):
        question = "今天是几号了"
        model = "glm-4-0520"
        tools = [CurrentContextToolDesc]
        response = chat(model=model, messages=question, tools=tools, stream=True, temperature=0.)
        content = show_response(response)
        self.assertIsNotNone(response.tool_calls)
        self.assertGreater(len(response.tool_calls), 0)
        self.assertEquals(CurrentContextToolDesc.content_resp, content)
        self.assertEquals("current_context", response.tool_calls[0].name)
        for tool_call in response.tool_calls:
            resp = on_tool_call(tool_call)
            logger.info(f"tool call resp:{resp}")
        response = chat(model=model, messages=question, tools=tools, tool_calls=response.tool_calls, stream=True, temperature=0.)
        assert len(response.tool_calls) == 0
        content = show_response(response)

        question = "世界上面积第三大的国家是哪个？"
        model = "glm-4-0520"
        tools = [CurrentContextToolDesc]
        # tools = []
        response = chat(model=model, messages=question, tools=tools, stream=True, temperature=0.)
        self.assertEqual(0, len(response.tool_calls))
        content = show_response(response)
        # logger.info("content:\n"+content)
        self.assertTrue("中国" in content)

    def test_embd(self):
        tests = ["你好", "hello"]
        embds = embedding(texts=tests, model="embedding-3", batch_size=2, dimensions=512)
        show_embeds(embds)
        self.assertEquals(2, len(embds))
        self.assertEquals(512, len(embds[0]))
        sim = get_embd_similarity(*embds)
        logger.info(f"{sim=:.2f}")
        self.assertGreaterEqual(sim, 0.5)
