from loguru import logger

from liteai.utils import set_logger
from tests.base import BasicTestCase


class TestDoubao(BasicTestCase):
    @classmethod
    def setUpClass(cls):
        set_logger(__name__)
        logger.info("start doubao test job")

    def test_basic_llm(self):
        super().basic_llm("doubao-lite-4k")
