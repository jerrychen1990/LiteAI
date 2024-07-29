#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/07/29 11:20:33
@Author  :   ChenHao
@Description  :   配置信息
@Contact :   jerrychen1990@gmail.com
'''


import os

LITEAI_ENV = os.environ.get("LITEAI_ENV", "dev")

LITEAI_HOME = os.environ.get("LITEAI_HOME", os.path.dirname(os.path.abspath(__file__)))
LOG_HOME = os.environ.get("LOG_HOME", os.path.join(LITEAI_HOME, "logs"))
