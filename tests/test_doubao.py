
from liteai.core import Message
from liteai.api import chat
from liteai.utils import set_logger, show_response
from loguru import logger

from tests.base import BasicTestCase


class TestDoubao(BasicTestCase):
    @classmethod
    def setUpClass(cls):
        set_logger(__name__)
        logger.info("start doubao test job")

    def test_basic_llm(self):
        super().basic_llm("doubao-lite-4k")
