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
from liteai.core import Message, ModelCard, ModelResponse, ToolCall, ToolDesc, Voice
from liteai.utils import truncate_dict_strings
from snippets import jdumps, retry, multi_thread


class BaseProvider:
    key: str = None
    allow_kwargs = None
    api_key_env = None

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get(self.api_key_env)
        # logger.debug(f"{self.api_key=}")
        if not self.api_key and not getattr(self, "base_url"):
            raise ValueError(f"api_key is required or set {self.api_key_env} in environment variables or set base_url")

    def pre_process(self, model: ModelCard, messages: List[Message], tools: List[ToolDesc],
                    tool_calls: List[ToolCall], stream: str, **kwargs) -> Tuple[List[dict], List[dict], dict]:
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

        # logger.debug(f"{messages=}")

        if not model.support_system:
            self._handle_system(model, messages, **kwargs)
        # logger.debug(f"{messages=}")

        tools = [tool.model_dump(exclude_none=True) for tool in tools]
        for tool_call in tool_calls:
            if tool_call.resp:
                tool_call_message = dict(role="tool", content=jdumps(tool_call.resp), tool_call_id=tool_call.tool_call_id)
                logger.debug(f"adding tool_call_message: {tool_call_message}")
                messages.append(tool_call_message)

        return messages, tools, new_kwargs

    def _handle_system(self, model: ModelCard, messages: List[dict], **kwargs) -> List[dict]:
        system = None
        last_user_message = None
        for message in messages:
            if message["role"] == "system":
                system = message
            if message["role"] == "user":
                last_user_message = message
        if system and last_user_message:
            logger.warning(f"model:{model.name} not support system, merge system message to last user message")
            messages.remove(system)
            if kwargs.get("handle_system", True):
                last_user_message["content"] = system["content"] + "\n" + last_user_message["content"]
        return messages

    @abstractmethod
    def post_process(self, response, **kwargs) -> ModelResponse:
        raise NotImplementedError

    @abstractmethod
    def post_process_stream(self, response) -> ModelResponse:
        raise NotImplementedError

    @abstractmethod
    def _inner_complete_(self, model: str, messages: List[dict], tools: List[ToolDesc], stream: bool, **kwargs) -> Any:
        raise NotImplementedError

    def complete(self, model: ModelCard, messages: List[Message], stream: bool,
                 tools: List[ToolDesc] = [], tool_calls: List[ToolCall] = [], **kwargs) -> ModelResponse:
        messages, dict_tools, kwargs = self.pre_process(model=model, messages=messages, tools=tools, stream=stream, tool_calls=tool_calls, ** kwargs)

        self.show_calling_info(messages, dict_tools, model, stream, **kwargs)
        response = self._inner_complete_(model.name, messages, stream=stream, tools=dict_tools, **kwargs)
        if stream:
            resp = self.post_process_stream(response, **kwargs)
        else:
            resp = self.post_process(response, **kwargs)
        resp = self.on_tool_call(resp, tools)
        return resp

    def on_tool_call(self, response: ModelResponse, tools: List[ToolDesc]) -> ModelResponse:
        tool_calls = []
        tools2desc = {tool.name: tool for tool in tools}
        tool_content = ""
        # logger.debug(f"on tool call for {len(response.tool_calls)} tools")
        for tool_call in response.tool_calls:
            if tool_call.name not in tools2desc:
                continue
            tool_desc: ToolDesc = tools2desc[tool_call.name]
            tool_call.tool_desc = tool_desc
            if tool_desc.content_resp:
                tool_content = tool_desc.content_resp
            # if tool_desc.is_inner:
                # logger.debug(f"calling inner tool {tool_call.name}")
                # tool_call.resp = tool_desc.func(**tool_call.parameters)
            tool_calls.append(tool_call)
        response.tool_calls = tool_calls
        logger.debug(f"get {len(tool_calls)} valid tool calls")
        if tool_content:
            logger.debug(f"set tool content {tool_content} to response")
            response.content = tool_content if isinstance(response.content, str) else (tool_content)
        return response

    def show_calling_info(self, messages: List[Message], tools: List[dict], model: ModelCard, stream: bool, **kwargs):
        show_messages = messages
        show_messages = truncate_dict_strings(messages, 50, key_pattern=["url"])

        message_show_type = kwargs.get("message_show_type", "human")
        logger.debug(f"message_show_type={message_show_type}")
        if message_show_type == "origin":
            message_str = str(show_messages)
        else:
            message_str = ""
            for idx, message in enumerate(show_messages, start=1):
                message_str += f"[{idx}].<{message['role']}>\n{message['content']}\n"

        calling_detail = f"calling {self.key} api with {model.name=}, {stream = }\nmessages=\n{message_str}"

        if tools:
            calling_detail += f"\ntools={jdumps(tools)}"
        if hasattr(self, "base_url"):
            calling_detail += f"\nbase_url = {getattr(self, 'base_url')}"
        calling_detail += f"\nkwargs = {jdumps(kwargs)}"
        logger.debug(calling_detail)

    def tts(self, text: str, model: ModelCard, stream: bool, **kwargs) -> Voice:
        raise Exception(f"provider {self.__class__.__name__} not support tts!")

    def asr(self, voice: Voice, model: ModelCard, **kwargs) -> str:
        raise Exception(f"provider {self.__class__.__name__} not support asr!")

    def embedding(self, texts: str | List[str], model: ModelCard, batch_size=8, **kwargs) -> List[List[float]] | List[float]:
        batch_func = multi_thread(work_num=batch_size, return_list=True)(self._embedding_single)
        return batch_func(data=texts, model=model, **kwargs)

    def _embedding_single_try(self, text: str, model: str, **kwargs) -> List[float]:
        raise Exception(f"provider {self.__class__.__name__} not support embedding!")

    def _embedding_single(self, text: str, model: ModelCard, retry_num=2, wait_time=(1, 2), **kwargs) -> List[float]:
        if retry_num:
            attempt = retry(retry_num=retry_num, wait_time=wait_time)(self._embedding_single_try)
        else:
            attempt = self._embedding_single_try
        return attempt(text=text, model=model.name, **kwargs)
