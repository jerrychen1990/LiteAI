
import os
from cozepy import ChatEventType, Coze, TokenAuth, Message as CozeMessage, ChatStatus, COZE_CN_BASE_URL
from loguru import logger
from openai import OpenAI
from snippets import add_callback2gen

from liteai.core import Message, ModelCard, ModelResponse, ToolDesc, Usage
from liteai.provider.base import BaseProvider
from liteai.utils import acc_chunks, get_text_chunk, image2base64

BASE_URL = os.environ.get("COZE_API_BASE", COZE_CN_BASE_URL)


class CozeProvider(BaseProvider):
    key: str = "coze"
    allow_kwargs = {"stream", "temperature", "top_p", "max_tokens", "user_id"}
    api_key_env = "COZE_API_KEY"

    def __init__(self, api_key: str = None, base_url: str = BASE_URL):
        self.base_url = base_url if base_url else BASE_URL
        super().__init__(api_key=api_key)
        # logger.debug(f"{self.api_key=}")
        self.client = Coze(auth=TokenAuth(token=self.api_key), base_url=self.base_url)

    def _check_coze_kwargs(self, **kwargs):
         if not kwargs.get("user_id"):
            raise ValueError("user_id is required")
    def _inner_complete_(self, model: str, messages: list[dict], stream: bool, tools: list[dict], **kwargs) -> any:
        self._check_coze_kwargs(**kwargs)

        additional_messages = [CozeMessage.build_user_question_text(msg["content"]) for msg in messages]
        logger.debug(f"{additional_messages=}")

        # print(f"msg: {msg}")
        bot_id = model.split("_")[-1]
        user_id = kwargs.get("user_id")
        logger.debug(f"{bot_id=}, {user_id=}")
        logger.debug(f"{self.client._base_url=}")

        if not stream:
            chat = self.client.chat.create(
                    additional_messages=additional_messages,
                    user_id=user_id,
                    bot_id=bot_id,
                )
            while chat.status == ChatStatus.IN_PROGRESS:
                chat = self.client.chat.retrieve(
                    conversation_id=chat.conversation_id,
                    chat_id=chat.id
                )
                if chat.status == ChatStatus.COMPLETED:
                    break
            messages = self.client.chat.messages.list(
                conversation_id=chat.conversation_id,
                chat_id=chat.id
            )
            return messages


        else:
            chat = self.client.chat.stream(
                additional_messages=additional_messages,
                user_id=user_id,
                bot_id=bot_id,
            )
            return chat


    def post_process(self, response, **kwargs) -> ModelResponse:
        # logger.debug(f"{response=}")
        for msg in response:
            if msg.role == "assistant" and not msg.content.startswith("{"):
                content = msg.content
                break
        usage = None
        return ModelResponse(content=content, usage=usage)

    def post_process_stream(self, response, **kwargs) -> ModelResponse:
        # gen = (e for e in (get_text_chunk(chunk) for chunk in response) if e)
        def gen():
            is_first_reasoning_content = True
            is_first_content = True
            for event in response:
                if event.event == ChatEventType.CONVERSATION_MESSAGE_DELTA:
                    if event.message.reasoning_content:
                        if is_first_reasoning_content:
                            is_first_reasoning_content = not is_first_reasoning_content
                            # print("----- reasoning_content start -----\n> ", end="", flush=True)
                        # print(event.message.reasoning_content, end="", flush=True)
                    else:
                        if is_first_content and not is_first_reasoning_content:
                            is_first_content = not is_first_content
                            print("----- reasoning_content end -----")
                        yield event.message.content

                if event.event == ChatEventType.CONVERSATION_CHAT_COMPLETED:
                    pass

        gen = add_callback2gen(gen(), acc_chunks)
        return ModelResponse(content=gen)




if __name__ == "__main__":
    provider = CozeProvider
    response = provider.complete(model="coze_7459590969592725516", messages=[Message(role="user", content="你好")], stream=False)
    print(response)
