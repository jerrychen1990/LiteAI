#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/06/25 16:39:05
@Author  :   ChenHao
@Description  :   
@Contact :   jerrychen1990@gmail.com
'''
from typing import Any, List, Tuple

from loguru import logger

from liteai.core import BaseProvider, ModelResponse, Message, Usage
from zhipuai import ZhipuAI

from liteai.utils import image2base64


class ZhipuProvider(BaseProvider):
    key: str = "zhipu"
    allow_kwargs = {"do_sample", "stream", "temperature", "top_p", "max_tokens"}
    api_key_env = "ZHIPUAI_API_KEY"

    def __init__(self, api_key: str = None):
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

    def _inner_complete_(self, model, messages: List[dict], stream: bool, ** kwargs) -> Any:
        # logger.debug(f"{self.client.api_key=}")
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream,
            **kwargs
        )
        return response

    def post_process(self, response) -> ModelResponse:
        content = response.choices[0].message.content
        usage = Usage(**response.usage.model_dump())
        return ModelResponse(content=content, usage=usage)

    def post_process_stream(self, response) -> ModelResponse:
        # for item in response:
        #     logger.debug(f"{item=}")
        #     # content = item.choices[0].message.content

        def _gen():
            acc = []
            for chunk in response:
                # logger.debug(f"{chunk=}")
                choices = chunk.choices
                if choices:
                    choice = choices[0]
                    if choice.delta.content:
                        delta_content = choice.delta.content
                        # logger.info(f"{delta_content}")
                        yield delta_content
                        acc.append(delta_content)
                _finish_reason = choice.finish_reason
                if _finish_reason:
                    if _finish_reason == "sensitive":
                        logger.warning(f"zhipu api finish with reason {_finish_reason}")
                        msg = "系统检测到输入或生成内容可能包含不安全或敏感内容，请您避免输入易产生敏感内容的提示语，感谢您的配合。"
                        acc.append(msg)
                        yield msg

            resp_msg = "".join(acc).strip()
            logger.debug(f"model generate answer:{resp_msg}")
        return ModelResponse(content=_gen())


if __name__ == "__main__":
    provider = ZhipuProvider()
    messages = [Message(role="user", content="你好")]
    resp = provider.complete(messages=messages, model="glm-3-turbo", stream=False)
    print(resp.content)
