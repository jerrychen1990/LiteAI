

from liteai.utils import set_logger
from loguru import logger

from tests.base import BasicTestCase


class TestTGI(BasicTestCase):

    @classmethod
    def setUpClass(cls):
        set_logger(__name__)
        logger.info("start test tgi job")

    def test_basic_llm(self):
        super().basic_llm(model="tgi_glm3_xz",
                          base_url="http://hz-model.bigmodel.cn/servyou-api")
