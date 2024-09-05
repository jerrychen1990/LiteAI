#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/07/31 15:50:42
@Author  :   ChenHao
@Description  :   
@Contact :   jerrychen1990@gmail.com
'''

from enum import Enum
from typing import Any, Dict, Iterable, List, Optional
from litellm import ConfigDict
from pydantic import Field
from typing import Optional
import warnings

import pydantic

warnings.filterwarnings("ignore", message=".*protected_namespaces.*")


class BaseModel(pydantic.BaseModel):
    model_config = ConfigDict(protected_namespaces=())


class Message(BaseModel):
    role: str = Field(description="The role of the sender in the conversation.", pattern="^(system|user|assistant)$", example="user")
    content: str | Iterable[str] = Field(description="The content of the message.")
    image: Optional[str] = Field(description="local path of image", default=None)


class Usage(BaseModel):
    prompt_tokens: Optional[int] = Field(description="输入token数量", default=None)
    completion_tokens: Optional[int] = Field(description="输出token数量", default=None)
    total_tokens: Optional[int] = Field(description="输入输出token数量总和", default=None)

    @property
    def total_tokens(self):
        return self.total_tokens if self.total_tokens is not None else self.prompt_tokens + self.completion_tokens


class LLMGenConfig(BaseModel):
    temperature: float = Field(description="temperature", default=0.7)
    max_tokens: Optional[int] = Field(description="max_tokens", default=None)


class Parameter(BaseModel):
    name: str = Field(description="参数名称")
    type: str = Field(description="参数类型")
    description: str = Field(description="参数描述")
    required: bool = Field(description="是否必填")


class ToolDesc(BaseModel):

    name: str = Field(..., description="工具名称")
    description: str = Field(..., description="工具描述")
    parameters: List[Parameter] = Field(..., description="工具参数")
    content_resp: Optional[str] = Field(description="调用工具时，返回的文字内容", default=None)
    is_local: bool = Field(description="是否本地工具", default=False)
    is_inner: bool = Field(description="是否在回答过程中需要执行的函数", default=False)

    def to_markdown(self):
        tool_info = f"**[名称]**:{self.name}\n\n**[描述]**:{self.description}\n\n"
        param_infos = []
        for parameter in self.parameters:
            param_infos.append(
                f"- **[参数名]**:{parameter.name}\n\n **[类型]**:{parameter.type}\n\n **[描述]**:{parameter.description}\n\n **[必填]**:{parameter.required}")
        param_infos = "\n\n".join(param_infos)
        return tool_info + param_infos


class ToolCall(BaseModel):
    tool_call_id: str = Field(description="工具调用ID,用于跟踪调用链")
    name: str = Field(description="工具名称")
    tool_desc: Optional[ToolDesc] = Field(description="工具描述", default=None)
    parameters: Dict[str, Any] = Field(description="工具调用的参数")
    extra_info: dict = Field(description="额外的信息", default=dict())
    resp: Dict = Field(description="工具执行的返回结果,为执行时为None", default=None)

    def to_markdown(self):
        return f"**[调用工具]**: {self.name} **[参数]**: {self.parameters} **[返回结果]**: {self.resp}"

    def to_assistant_message(self):
        return f"调用工具: {self.name}, 参数: {self.parameters}, 返回结果: {self.resp}"


class ModelResponse(BaseModel):
    content: Optional[str | Iterable[str]] = Field(description="模型的回复，字符串或者生成器", default=None)
    image: Optional[str] = Field(description="图片URL", default=None)
    # tool_calls: Optional[list[ToolCall]] = Field(description="工具调用列表", default=list())
    usage: Optional[Usage] = Field(description="token使用情况", default=None)
    # perf: Optional[Perf] = Field(description="性能指标", default=None)
    tool_calls: Optional[List[ToolCall]] = Field(description="工具调用列表", default=[])
    details: Optional[dict] = Field(description="请求模型的细节信息", default=dict())

    class Config:
        protected_namespaces = ()


class Voice(BaseModel):
    byte_stream: Iterable[bytes] | bytes = Field(description="音频字节流")
    file_path: Optional[str] = Field(description="文件路径", default=None)


class ModelType(str, Enum):
    LLM = "llm"
    TTS = "tts"
    EMBEDDING = "embedding"


class ModelCard(BaseModel):
    name: str = Field(description="模型名称")
    model_type: ModelType = Field(description="模型类型", default=ModelType.LLM)
    description: str = Field(description="模型描述", default="")
    support_system: bool = Field(description="是否支持系统对话", default=True)
    support_vision: bool = Field(description="是否支持图像输入", default=False)
    provider: str = Field(description="模型调用器key", default=None)
