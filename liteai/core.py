from abc import abstractmethod
import os
from typing import Any, List, Optional, Tuple
from pydantic import BaseModel, Field
from loguru import logger
from typing import Any, Generator, List, Optional

from liteai.utils import truncate_dict_strings


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
    api_key_env = None

    def __init__(self, api_key: str = None):
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.environ.get(self.api_key_env)
        logger.debug(f"{self.api_key=}")
        if not self.api_key:
            raise ValueError(f"api_key is required or set {self.api_key_env} in environment variables")

    def pre_process(self, model: str, messages: List[Message], stream: str, **kwargs) -> Tuple[List[dict], dict]:
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
            self._handle_system(model, messages)

        return messages, new_kwargs

    def _support_system(self, model: str):
        return True

    def _handle_system(self, model:str, messages: List[dict]) -> List[dict]:
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
            last_user_message["content"] = system["content"] + "\n" + last_user_message["content"]
        return messages

    @abstractmethod
    def post_process(self, response) -> ModelResponse:
        raise NotImplementedError

    @abstractmethod
    def post_process_stream(self, response) -> ModelResponse:
        raise NotImplementedError

    @abstractmethod
    def _inner_complete_(self, model, messages: List[dict], stream: bool, **kwargs) -> Any:
        raise NotImplementedError

    def complete(self, model, messages: List[Message], stream: bool, **kwargs) -> ModelResponse:

        messages, kwargs = self.pre_process(model, messages, stream, **kwargs)
        show_message = truncate_dict_strings(messages, 50)
        logger.debug(f"calling {self.key} api with {model=}, messages={show_message}, {stream=}, {kwargs=}")
        response = self._inner_complete_(model, messages, stream=stream, **kwargs)
        if stream:
            return self.post_process_stream(response)
        else:
            return self.post_process(response)
