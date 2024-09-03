
from liteai.core import Message
from liteai.api import chat
from liteai.utils import set_logger, show_response
from loguru import logger

from tests.base import BasicTestCase


class TestQwen(BasicTestCase):
    @classmethod
    def setUpClass(cls):
        set_logger(__name__)
        logger.info("start test qwen job")

    def test_basic_llm(self):
        super().basic_llm("qwen-turbo")

    def test_vision_chat(self):
        question = "这张图里有什么?"
        image_path = "./data/Pikachu.png"
        model = "qwen-vl-plus"
        messages = [Message(role="user", content=question, image=image_path)]
        resp = chat(model=model, messages=messages, stream=False, temperature=0.)
        show_response(resp)
