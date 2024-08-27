#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/08/06 10:05:51
@Author  :   ChenHao
@Description  :   
@Contact :   jerrychen1990@gmail.com
'''


import json
import os
from typing import List
from litellm import Tuple
from loguru import logger
import requests
from liteai.core import Message, ModelCard, ToolDesc, Voice
from liteai.provider.open_ai import OpenAIProvider
from liteai.voice import build_voice
from snippets import jdumps


class MinimaxProvider(OpenAIProvider):
    key: str = "minimax"
    allow_kwargs = {"do_sample", "stream", "temperature", "top_p", "max_tokens"}
    api_key_env = "MINIMAX_API_KEY"

    def __init__(self, api_key: str = None, **kwargs):
        super().__init__(api_key=api_key, base_url="https://api.minimax.chat/v1")

    def pre_process(self, model: ModelCard, messages: List[Message], tools: List[ToolDesc], stream: bool, **kwargs) -> Tuple[List[dict], dict]:
        if kwargs.get("temperature") == 0.:
            logger.debug(f"provider {self.key} not support temperature=0, setting temperature to 0.0001")
            kwargs["temperature"] = 0.0001
        messages, tools, kwargs = super().pre_process(model=model, messages=messages, tools=tools, stream=stream, **kwargs)
        return messages, tools, kwargs

    def tts(self, text: str, model: ModelCard, version="t2a_v2", tgt_path: str = None,
            append=False, stream=False, voice_id="tianxin_xiaoling", speed=1, pitch=0) -> Voice:

        group_id = os.environ["MINIMAX_GROUP_ID"]
        url = f"https://api.minimax.chat/v1/{version}?GroupId={group_id}"

        payload = {
            "model": model.name,
            "text": text,
            "stream": stream,
            "voice_setting": {
                "voice_id": voice_id,
                "speed": speed,
                "vol": 1,
                "pitch": pitch
            },
            "audio_setting": {
                "sample_rate": 32000,
                "bitrate": 128000,
                "format": "mp3",
                "channel": 1
            }
        }
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        logger.debug(f"calling minimax tts url:{url} with payload:\n{jdumps(payload)}")
        response = requests.request("POST", url, headers=headers, stream=stream, json=payload)
        logger.debug(f"{response.status_code=}")
        response.raise_for_status()
        if not stream:
            data = response.json()
            if "data" not in data:
                logger.exception(data)
                raise Exception(data)
            byte_stream = bytes.fromhex(data['data']['audio'])
        else:
            def _get_bytes(chunk):
                if chunk:
                    if chunk[:5] == b'data:':
                        data = json.loads(chunk[5:])
                        if "data" in data and "extra_info" not in data:
                            if "audio" in data["data"]:
                                audio = data["data"]['audio']
                                return bytes.fromhex(audio)
            byte_stream = (e for e in (_get_bytes(chunk) for chunk in response.raw) if e)
        voice = build_voice(byte_stream=byte_stream, file_path=tgt_path)
        return voice


if __name__ == "__main__":
    provider = MinimaxProvider()
    messages = [Message(role="user", content="你好")]
    resp = provider.complete(messages=messages, model="abab6.5s-chat", stream=False)
    print(resp.content)
