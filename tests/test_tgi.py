import unittest


from liteai.core import Message
from liteai.api import chat
from liteai.utils import set_logger, show_response
from loguru import logger


class TestTGI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_logger(__name__)
        logger.info("start tgi job")

    @unittest.skip("skip tgi test")
    def test_local_tgi(self):
        system = "用英文回答我的问题, 80个单词以内"
        question = "列出国土面积最大的五个国家"
        model = "tgi_glm3_32b"
        base_url = "http://hz-model.bigmodel.cn/servyou-api"
        # litellm.set_verbose = True
        # os.environ['LITELLM_LOG'] = 'DEBUG'

        messages = [Message(role="system", content=system),
                    Message(role="user", content=question)]
        response = chat(model=model, messages=messages, stream=False, base_url=base_url, temperature=0.01, max_tokens=1024)
        show_response(response)

        messages.extend([Message(role="assistant", content=response.content),
                        Message(role="user", content="介绍第二个")])
        response = chat(model=model, messages=messages, stream=True, base_url=base_url, temperature=0., max_tokens=1024)
        show_response(response)
        # self.assertTrue("Canada" in response.content)
