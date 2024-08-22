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

from liteai.core import Parameter, ToolDesc


class Tool(BaseModel):
    tool_desc: ToolDesc = Field(..., description="工具描述")
    callable: Callable = Field(..., description="工具执行函数")

    def execute(self, *args, **kwargs) -> Any:
        return self.callable(*args, **kwargs)


def get_current_context(time_type: str):
    import datetime
    import time
    fmt = '%Y-%m-%d %H:%M:%S' if time_type == 'datetime' else '%Y-%m-%d'
    current_time = datetime.datetime.fromtimestamp(time.time()).strftime(fmt)
    return dict(current_time=current_time)


CurrentContextToolDesc = ToolDesc(name="current_context", description="获取当前时间",
                                  parameters=[Parameter(name="time_type", description="时间格式，分为两种datetime:精确到秒, date:精确到日",
                                                        type="string", required=True)],
                                  content_resp="正在帮您获取当前时间...")

CURRENT_CONTEXT_TOOL = Tool(tool_desc=CurrentContextToolDesc, callable=get_current_context)

if __name__ == "__main__":
    resp = CURRENT_CONTEXT_TOOL.execute(time_type="datetime")
    logger.info(resp)
