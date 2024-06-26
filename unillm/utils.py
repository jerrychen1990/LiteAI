#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/06/25 17:44:35
@Author  :   ChenHao
@Description  :   
@Contact :   jerrychen1990@gmail.com
'''
import os
import sys
from loguru import logger
from .core import ModelResponse


def show_response(response: ModelResponse):
    content = response.content
    if isinstance(content, str):
        logger.info(content)
    else:
        for item in content:
            logger.info(item)


def set_logger(module):
    print(f"setting logger for {module=}")
    if 0 in logger._core.handlers:
        logger.remove(0)
    UNILLM_ENV = os.environ.get("UNILLM_ENV", "DEV")

    dev_fmt = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> [<level>{level: <8}</level>] - <cyan>{file}</cyan>:<cyan>{line}</cyan>[<cyan>{name}</cyan>:<cyan>{function}</cyan>] - <level>{message}</level>"
    if UNILLM_ENV.upper() == "DEV":
        logger.add(sys.stdout, level="DEBUG", filter=lambda r: module in r["name"], format=dev_fmt, enqueue=True, colorize=True)
