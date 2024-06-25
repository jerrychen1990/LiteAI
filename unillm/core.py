from abc import abstractmethod
from enum import Enum
from typing import Any, List, Optional, Tuple
from litellm import ModelResponse, CustomStreamWrapper
from pydantic import BaseModel, Field
from loguru import logger


class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"


class Message(BaseModel):
    role: str = Field(description="The role of the sender in the conversation.", pattern="^(system|user|assistant)$", example="user")
    content: str = Field(description="The content of the message.")
    image: Optional[str] = Field(description="The image attached to the message. base64", default=None)


class BaseProvider:
    key: str = None

    def pre_process(self, messages: List[Message], **kwargs) -> Tuple[List[dict], dict]:
        messages = [message.model_dump(exclude_none=True) for message in messages]
        return messages, kwargs

    def post_process(self, response) -> ModelResponse:
        return response

    def post_process_stream(self, response) -> CustomStreamWrapper:
        return response

    @abstractmethod
    def _inner_complete_(self, model, messages: List[dict], **kwargs) -> Any:
        raise NotImplementedError

    def complete(self, model, messages: List[Message], stream: bool, **kwargs) -> ModelResponse | CustomStreamWrapper:

        messages, kwargs = self.pre_process(messages, **kwargs)
        logger.debug(f"calling {self.key} api with {messages=}, {kwargs=}")
        response = self._inner_complete_(model, messages, **kwargs)
        if stream:
            return self.post_process_stream(response)
        else:
            return self.post_process(response)
