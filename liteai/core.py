#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/07/31 15:50:42
@Author  :   ChenHao
@Description  :   
@Contact :   jerrychen1990@gmail.com
'''

from typing import Optional
from pydantic import BaseModel, Field
from typing import Generator, Optional


# class Role(str, Enum):
#     system = "system"
#     user = "user"
#     assistant = "assistant"


class Message(BaseModel):
    role: str = Field(description="The role of the sender in the conversation.", pattern="^(system|user|assistant)$", example="user")
    content: str = Field(description="The content of the message.")
    image: Optional[str] = Field(description="local path of image", default=None)


class Usage(BaseModel):
    prompt_tokens: int = Field(description="输入token数量", default=None)
    completion_tokens: int = Field(description="输出token数量", default=None)
    # total_tokens: int = Field(description="输入输出token数量总和", default=None)


class ModelResponse(BaseModel):
    content: Optional[str | Generator] = Field(description="模型的回复，字符串或者生成器", default=None)
    image: Optional[str] = Field(description="图片URL", default=None)
    # tool_calls: Optional[list[ToolCall]] = Field(description="工具调用列表", default=list())
    usage: Optional[Usage] = Field(description="token使用情况", default=None)
    # perf: Optional[Perf] = Field(description="性能指标", default=None)
    details: Optional[dict] = Field(description="请求模型的细节信息", default=dict())
