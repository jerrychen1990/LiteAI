

import os
from liteai.config import DATA_HOME
from liteai.core import Voice
from liteai.api import asr
from liteai.utils import set_logger
from loguru import logger

from liteai.voice import file2voice
from tests.base import BasicTestCase


class TestXunfei(BasicTestCase):
    @classmethod
    def setUpClass(cls):
        set_logger(__name__)
        logger.info("start xunfei test job")

    def test_asr(self):
        voice: Voice = file2voice(os.path.join(DATA_HOME, "hello.mp3"))
        asr_test = asr(voice=voice, model="xunfei_asr", app_id="9b5cc933")
        logger.info(f"{asr_test=}")
        self.assertIn("天过得怎么样", asr_test)
