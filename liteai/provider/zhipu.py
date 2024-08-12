#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/06/25 16:39:05
@Author  :   ChenHao
@Description  :   
@Contact :   jerrychen1990@gmail.com
'''
import json
from typing import Any, List, Tuple

from loguru import logger


from liteai.core import ModelResponse, Message, ToolCall, ToolDesc, Usage
from zhipuai import ZhipuAI
from liteai.provider.base import BaseProvider
from snippets import add_callback2gen

from liteai.utils import get_chunk_data, image2base64, acc_chunks


def build_tool_calls(tool_calls) -> List[ToolCall]:
    # logger.debug(f"tool_calls: {tool_calls}")
    if tool_calls is None:
        return []
    rs = []
    for tool_call in tool_calls:
        parameters = json.loads(tool_call.function.arguments)

        tmp = ToolCall(tool_call_id=tool_call.id, name=tool_call.function.name, parameters=parameters)
        rs.append(tmp)
    # logger.debug(f"tool_calls: {rs}")
    return rs


class ZhipuProvider(BaseProvider):
    key: str = "zhipu"
    allow_kwargs = {"do_sample", "stream", "temperature", "top_p", "max_tokens", "meta"}
    api_key_env = "ZHIPUAI_API_KEY"

    def __init__(self, api_key: str = None, **kwargs):
        super().__init__(api_key=api_key)
        self.client = ZhipuAI(api_key=self.api_key)

    def _support_system(self, model: str):
        model = model.lower()
        if "glm-4" in model:
            return True
        return "chatglm3" in model or "glm-3" in model

    def pre_process(self, model: str, messages: List[Message], stream: bool, **kwargs) -> Tuple[List[dict], dict]:
        if kwargs.get("temperature") == 0.:
            del kwargs["temperature"]
            kwargs["do_sample"] = False
        messages, kwargs = super().pre_process(model, messages, stream, **kwargs)
        for message in messages:
            # logger.debug(f"{message=}")
            if message.get("image"):
                base64 = image2base64(message["image"])
                message["content"] = [dict(type="text", text=message["content"]),
                                      dict(type="image_url", image_url=dict(url=base64))]
                del message["image"]
        return messages, kwargs

    @classmethod
    def tool2zhipu_tool(cls, tool: ToolDesc):
        properties = {p.name: dict(description=p.description, type=p.type) for p in tool.parameters}
        required = [p.name for p in tool.parameters if p.required]
        parameters = dict(type="object", properties=properties, required=required)

        rs = dict(type="function", function=dict(name=tool.name, description=tool.description, parameters=parameters))
        return rs

    def _inner_complete_(self, model, messages: List[dict], stream: bool, tools: List[ToolDesc], **kwargs) -> Any:
        # logger.debug(f"{self.client.api_key=}")
        zhipu_tools = [self.tool2zhipu_tool(tool) for tool in tools]
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream,
            tools=zhipu_tools,
            **kwargs
        )
        return response

    def post_process(self, response) -> ModelResponse:
        logger.debug(f"{response=}")
        content = response.choices[0].message.content
        tool_calls = build_tool_calls(response.choices[0].message.tool_calls)

        usage = Usage(**response.usage.model_dump())
        return ModelResponse(content=content, usage=usage, tool_calls=tool_calls)

    def post_process_stream(self, response) -> ModelResponse:
        gen = (e for e in (get_chunk_data(chunk) for chunk in response) if e)
        gen = add_callback2gen(gen, acc_chunks)
        return ModelResponse(content=gen)


if __name__ == "__main__":
    provider = ZhipuProvider()
    messages = [Message(role="user", content="你好")]
    resp = provider.complete(messages=messages, model="glm-3-turbo", stream=False)
    print(resp.content)
