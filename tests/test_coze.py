

from loguru import logger

from liteai.api import chat
from liteai.utils import set_logger, show_response
from tests.base import BasicTestCase


class TestCoze(BasicTestCase):
    @classmethod
    def setUpClass(cls):
        set_logger(__name__)
        logger.info("start coze test job")

    def test_coze_agent(self):
        messages = "你好呀"
        response = chat(model="coze_7459590969592725516", messages=messages, stream=True, temperature=0.0, user_id="test_user")
        content = show_response(response)
        logger.info(f"content: {content}")




# if __name__ == "__main__":
#     messages = "你好呀"
#     response = chat(model="coze_7459590969592725516", messages=messages, stream=True, temperature=0.0, user_id="test_user")
#     content = show_response(response)
#     logger.info(f"content: {content}")

