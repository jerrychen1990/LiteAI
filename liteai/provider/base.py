#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/07/31 15:50:13
@Author  :   ChenHao
@Description  :   
@Contact :   jerrychen1990@gmail.com
'''

from abc import abstractmethod
import os
from typing import Any, List, Tuple
from loguru import logger
from typing import Any, List
from liteai.core import Message, ModelResponse, ToolDesc, Voice
from liteai.utils import truncate_dict_strings
from snippets import jdumps, retry, multi_thread


class BaseProvider:
    key: str = None
    allow_kwargs = None
    api_key_env = None

    def __init__(self, api_key: str = None):
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.environ.get(self.api_key_env)
        # logger.debug(f"{self.api_key=}")
        if not self.api_key and not getattr(self, "base_url"):
            raise ValueError(f"api_key is required or set {self.api_key_env} in environment variables or set base_url")

    def pre_process(self, model: str, messages: List[Message], tools: List[ToolDesc], stream: str, **kwargs) -> Tuple[List[dict], dict]:
        new_kwargs = dict()
        ignore_kwargs = dict()
        # logger.debug(f"{self.allow_kwargs=}")
        if self.allow_kwargs:
            for k, v in kwargs.items():
                if k in self.allow_kwargs:
                    new_kwargs[k] = v
                else:
                    ignore_kwargs[k] = v
        else:
            new_kwargs = dict(**kwargs)

        # logger.debug(f"{new_kwargs=}")
        # logger.debug(f"{ignore_kwargs=}")
        if ignore_kwargs:
            logger.warning(f"ignoring {len(ignore_kwargs)} unknown kwargs: {ignore_kwargs}")

        messages = [message.model_dump(exclude_none=True) for message in messages]
        if not self._support_system(model):
            self._handle_system(model, messages, **kwargs)

        tools = [tool.model_dump(exclude_none=True) for tool in tools]

        return messages, tools, new_kwargs

    def _support_system(self, model: str):
        return True

    def _handle_system(self, model: str, messages: List[dict], **kwargs) -> List[dict]:
        system = None
        last_user_message = None
        for message in messages:
            if message["role"] == "system":
                system = message
            if message["role"] == "user":
                last_user_message = message
        if system and last_user_message:
            logger.warning(f"model:{model} not support system, merge system message to last user message")
            messages.remove(system)
            if kwargs.get("handle_system", True):
                last_user_message["content"] = system["content"] + "\n" + last_user_message["content"]
        return messages

    @abstractmethod
    def post_process(self, response) -> ModelResponse:
        raise NotImplementedError

    @abstractmethod
    def post_process_stream(self, response) -> ModelResponse:
        raise NotImplementedError

    @abstractmethod
    def _inner_complete_(self, model, messages: List[dict], tools: List[ToolDesc], stream: bool, **kwargs) -> Any:
        raise NotImplementedError

    def complete(self, model, messages: List[Message], stream: bool, tools: List[ToolDesc] = [], **kwargs) -> ModelResponse:

        messages, tools, kwargs = self.pre_process(model, messages, tools, stream, **kwargs)
        show_message = messages
        show_message = truncate_dict_strings(messages, 50, key_pattern=["url"])
        calling_detail = f"calling {self.key} api with {model = }, {stream = }\nkwargs = {jdumps(kwargs)}\nmessages = {jdumps(show_message)}"
        if tools:
            calling_detail += f"\ntools={jdumps(tools)}"
        if hasattr(self, "base_url"):
            calling_detail += f"\nbase_url = {getattr(self, 'base_url')}"
        logger.debug(calling_detail)
        response = self._inner_complete_(model, messages, stream=stream, tools=tools, **kwargs)
        if stream:
            return self.post_process_stream(response)
        else:
            return self.post_process(response)

    def tts(self, text: str, model: str, stream: bool, **kwargs) -> Voice:
        raise Exception(f"provider {self.__class__.__name__} not support tts!")

    def embedding(self, texts: str | List[str], model: str, batch_size=8, **kwargs) -> List[List[float]] | List[float]:
        batch_func = multi_thread(work_num=batch_size, return_list=True)(self._embedding_single)
        return batch_func(data=texts, model=model, **kwargs)

    def _embedding_single_try(self, text: str, model: str, **kwargs) -> List[float]:
        raise Exception(f"provider {self.__class__.__name__} not support embedding!")

    def _embedding_single(self, text: str, model: str, retry_num=2, wait_time=(1, 2), **kwargs) -> List[float]:
        if retry_num:
            attempt = retry(retry_num=retry_num, wait_time=wait_time)(self._embedding_single_try)
        else:
            attempt = self._embedding_single_try
        return attempt(text=text, model=model, **kwargs)
