#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/06/25 16:39:05
@Author  :   ChenHao
@Description  :   
@Contact :   jerrychen1990@gmail.com
'''
import os
from typing import Any, List, Tuple
from ..core import BaseProvider, ModelResponse, Message
from zhipuai import ZhipuAI


class ZhipuProvider(BaseProvider):
    key: str = "zhipu"

    def __init__(self, api_key: str):
        self.api_key = api_key or os.environ["ZHIPU_API_KEY"]
        self.client = ZhipuAI(api_key=api_key)

    def pre_process(self, messages: List[Message], **kwargs) -> Tuple[List[dict], dict]:
        kwargs.get("temperature") == 0.
        kwargs["temperature"] = 0.1
        kwargs["do_sample"] = False
        return super().pre_process(messages, **kwargs)

    def _inner_complete_(self, model, messages: List[dict], **kwargs) -> Any:
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
        return response

    def post_process(self, response) -> ModelResponse:
        model_response = ModelResponse(**response.model_dump())
        return model_response
