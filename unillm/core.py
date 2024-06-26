from abc import abstractmethod
from typing import Any, List, Optional, Tuple
from pydantic import BaseModel, Field
from loguru import logger
from typing import Any, Generator, List, Optional

from unillm.utils import truncate_dict_strings


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


class BaseProvider:
    key: str = None
    allow_kwargs = None

    def pre_process(self, messages: List[Message], **kwargs) -> Tuple[List[dict], dict]:
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
        return messages, new_kwargs

    @abstractmethod
    def post_process(self, response) -> ModelResponse:
        raise NotImplementedError

    @abstractmethod
    def post_process_stream(self, response) -> ModelResponse:
        raise NotImplementedError

    @abstractmethod
    def _inner_complete_(self, model, messages: List[dict], **kwargs) -> Any:
        raise NotImplementedError

    def complete(self, model, messages: List[Message], stream: bool, **kwargs) -> ModelResponse:

        messages, kwargs = self.pre_process(messages, **kwargs)
        show_message = truncate_dict_strings(messages, 50)
        logger.debug(f"calling {self.key} api with messages={show_message}, {kwargs=}")
        response = self._inner_complete_(model, messages, stream=stream, **kwargs)
        if stream:
            return self.post_process_stream(response)
        else:
            return self.post_process(response)
