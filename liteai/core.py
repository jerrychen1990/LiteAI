#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/07/31 15:50:42
@Author  :   ChenHao
@Description  :   
@Contact :   jerrychen1990@gmail.com
'''

import os
from typing import Iterable, Optional
from loguru import logger
from pydantic import BaseModel, Field
from typing import Optional
from snippets.utils import add_callback2gen


class Message(BaseModel):
    role: str = Field(description="The role of the sender in the conversation.", pattern="^(system|user|assistant)$", example="user")
    content: str = Field(description="The content of the message.")
    image: Optional[str] = Field(description="local path of image", default=None)


class Usage(BaseModel):
    prompt_tokens: Optional[int] = Field(description="输入token数量", default=None)
    completion_tokens: Optional[int] = Field(description="输出token数量", default=None)
    total_tokens: Optional[int] = Field(description="输入输出token数量总和", default=None)

    @property
    def total_tokens(self):
        return self.total_tokens if self.total_tokens is not None else self.prompt_tokens + self.completion_tokens


class ModelResponse(BaseModel):
    content: Optional[str | Iterable[str]] = Field(description="模型的回复，字符串或者生成器", default=None)
    image: Optional[str] = Field(description="图片URL", default=None)
    # tool_calls: Optional[list[ToolCall]] = Field(description="工具调用列表", default=list())
    usage: Optional[Usage] = Field(description="token使用情况", default=None)
    # perf: Optional[Perf] = Field(description="性能指标", default=None)
    details: Optional[dict] = Field(description="请求模型的细节信息", default=dict())


def _dump_voice(voice_stream: Iterable[bytes], path: str):
    with open(path, "wb") as f:
        logger.debug(f"save voice to {path}")
        for chunk in voice_stream:
            if chunk is not None and chunk != b'\n':
                decoded_hex = chunk
                f.write(decoded_hex)


class Voice(BaseModel):
    byte_stream: Iterable[bytes] | bytes = Field(description="音频字节流")
    file_path: Optional[str] = Field(description="文件路径", default=None)

    def build_voice(byte_stream, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if file_path:
            if isinstance(byte_stream, bytes):
                logger.debug(f"save voice to {file_path}")
                with open(file_path, "wb") as f:
                    f.write(byte_stream)
            else:
                byte_stream = add_callback2gen(byte_stream, _dump_voice, path=file_path)
        return Voice(byte_stream=byte_stream, file_path=file_path)
