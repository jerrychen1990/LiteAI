import unittest

from loguru import logger

#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/08/08 17:19:44
@Author  :   ChenHao
@Description  :   测试公共函数
@Contact :   jerrychen1990@gmail.com
'''
from liteai.api import list_models


class TestCommon(unittest.TestCase):
    def test_list_model(self):
        models = list_models()
        logger.info(f"models: {models}")
