#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/06/25 17:44:35
@Author  :   ChenHao
@Description  :   
@Contact :   jerrychen1990@gmail.com
'''
from loguru import logger

from litellm import ModelResponse, CustomStreamWrapper


def show_response(response: ModelResponse | CustomStreamWrapper):
    if isinstance(response, ModelResponse):
        logger.info(response.choices[0].message.content)
