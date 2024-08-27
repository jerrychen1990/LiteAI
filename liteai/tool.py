#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/08/12 19:54:53
@Author  :   ChenHao
@Description  :   工具类
@Contact :   jerrychen1990@gmail.com
'''


from typing import Any, Callable

from litellm import Field
from loguru import logger
from openai import BaseModel

from liteai.core import Parameter, ToolCall, ToolDesc


class Tool(BaseModel):
    tool_desc: ToolDesc = Field(..., description="工具描述")
    callable: Callable = Field(..., description="工具执行函数")

    def execute(self, *args, **kwargs) -> Any:
        logger.debug(f"executing tool:{self.tool_desc.name} with function:{self.callable}, with args:{args}, kwargs:{kwargs}")
        return self.callable(*args, **kwargs)


def get_current_context(time_type: str):
    import datetime
    import time
    fmt = '%Y-%m-%d %H:%M:%S' if time_type == 'datetime' else '%Y-%m-%d'
    current_time = datetime.datetime.fromtimestamp(time.time()).strftime(fmt)
    today = datetime.date.today()
    weekday = today.weekday()
    days = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]

    chinese_weekday = days[weekday]
    return dict(current_time=current_time, weekday=chinese_weekday)


def on_tool_call(tool_call: ToolCall):
    logger.debug(f"handling {tool_call}")
    assert tool_call.tool_desc is not None
    if tool_call.tool_desc.is_inner:
        tool = NAME2TOOL.get(tool_call.tool_desc.name)
        if tool is None:
            logger.warning(f"tool {tool_call.tool_desc.name} not found")
        else:
            resp = tool.execute(**tool_call.parameters)
            tool_call.resp = resp
    return tool_call.resp


CurrentContextToolDesc = ToolDesc(name="current_context", description="获取当前环境，包括日期、星期几、当前时间",
                                  parameters=[Parameter(name="time_type", description="时间格式，分为两种datetime:精确到秒, date:精确到日",
                                                        type="string", required=True)],
                                  content_resp="正在帮您获取当前时间...",
                                  is_local=True, is_inner=True)
TOOL_DESCS = [CurrentContextToolDesc]
NAME2TOOL_DESC = {tool_desc.name: tool_desc for tool_desc in TOOL_DESCS}


CURRENT_CONTEXT_TOOL = Tool(tool_desc=CurrentContextToolDesc, callable=get_current_context)

TOOLS = [CURRENT_CONTEXT_TOOL]
NAME2TOOL = {tool.tool_desc.name: tool for tool in TOOLS}


__ALL__ = ["TOOL_DESCS", "TOOLS", "NAME2TOOL_DESC", "NAME2TOOL"]
__ALL__.extend(TOOL_DESCS)
__ALL__.extend(TOOLS)

if __name__ == "__main__":
    resp = CURRENT_CONTEXT_TOOL.execute(time_type="datetime")
    logger.info(resp)
