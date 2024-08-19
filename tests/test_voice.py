import unittest


from liteai.voice import *
from liteai.utils import set_logger
from loguru import logger


class TestVoice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_logger(__name__)
        logger.info("start test voice job")

    def test_file_voice(self):
        voice_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "hello.mp3")
        voice = file2voice(voice_path)
        play_voice(voice)
