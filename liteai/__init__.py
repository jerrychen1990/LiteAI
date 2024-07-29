#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/06/25 18:39:23
@Author  :   ChenHao
@Description  :   
@Contact :   jerrychen1990@gmail.com
'''

from liteai.config import LITEAI_ENV, LOG_HOME
from snippets import set_logger


set_logger(LITEAI_ENV, __name__, log_dir=LOG_HOME)
