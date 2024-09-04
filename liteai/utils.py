#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/06/25 17:44:35
@Author  :   ChenHao
@Description  :   
@Contact :   jerrychen1990@gmail.com
'''
import base64
from io import BytesIO
from typing import Iterable, List
from loguru import logger
from PIL import Image
import numpy as np
from liteai.config import LITEAI_ENV
from liteai.core import ModelResponse, ToolCall
from snippets import batchify
import snippets
from snippets.utils import jdumps


def show_response(response: ModelResponse, batch_size=10):
    content = response.content
    if response.tool_calls:
        for tool_call in response.tool_calls:
            logger.info(f"tool_call: {jdumps(tool_call.model_dump())}")

    if isinstance(content, str):
        logger.info("content:\n"+content)
        return content
    else:
        logger.info("stream content:")
        acc = ""
        for item in batchify(content, batch_size):
            chunk = "".join(item)
            logger.info(chunk)
            acc += chunk
        return acc


def show_embeds(embds: List[List[float]] | List[float], sample_num=2, sample_size=4):
    if isinstance(embds[0], list):
        logger.info(f"embd dim:{len(embds[0])}")
        for i, embd in enumerate(embds[:sample_num]):
            logger.info(f"embd[{i}]: {embd[:sample_size]}")
    else:
        logger.info(f"embd dim:{len(embds)}")
        logger.info(f"embd: {embds[:sample_size]}")


def set_logger(module_name, **kwargs):
    return snippets.set_logger(env=LITEAI_ENV, module_name=module_name, **kwargs)


def image2base64(image_path):
    # 打开图像文件
    with Image.open(image_path) as image:
        # 创建一个字节流对象
        buffered = BytesIO()
        # 将图像保存到字节流对象中
        image.save(buffered, format="PNG")
        # 获取字节流中的字节数据
        img_bytes = buffered.getvalue()
        # 将字节数据编码为Base64字符串
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return img_base64


def truncate_dict_strings(data: dict, max_length: int, key_pattern=None) -> dict:
    def truncate_string(s):
        return s if len(s) <= max_length else s[:max_length//2] + '...' + s[-max_length//2:]

    def process_item(key, item):
        if isinstance(item, dict):
            return {k: process_item(k, v) for k, v in item.items()}
        elif isinstance(item, list):
            return [process_item(key, i) for i in item]
        elif isinstance(item, str) and key in key_pattern:
            return truncate_string(item)
        else:
            return item

    return process_item(None, data)


def get_text_chunk(chunk):
    # logger.debug(f"{type(chunk)=}, {chunk=}")
    choices = chunk.choices
    if choices:
        choice = choices[0]
        # logger.debug(f"{choice=}")
        if choice.delta and choice.delta.content:
            delta_content = choice.delta.content
            # logger.info(f"{delta_content}")
            return delta_content


def extract_tool_calls(chunks: Iterable):
    tool_calls = []
    for chunk in chunks:
        # logger.debug(f"{chunk=}")
        if delta := chunk.choices[0].delta:
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    tool_call = ToolCall(name=tc.function.name, parameters=eval(tc.function.arguments), tool_call_id=tc.id)
                    tool_calls.append(tool_call)
        break
    return tool_calls


def acc_chunks(acc):
    resp_msg = "".join(acc).strip()
    logger.debug(f"model generate answer:{resp_msg}")


def get_embd_similarity(embd1: List[float], embd2: List[float]):
    embd1 = np.array(embd1)
    embd2 = np.array(embd2)
    sim = np.dot(embd1, embd2) / (np.linalg.norm(embd1) * np.linalg.norm(embd2))
    return sim


if __name__ == "__main__":
    data = {
        'name': 'Pikachu',
        'description': 'Pikachu is an Electric-type Pokémon introduced in Generation I.',
        'abilities': ['Static', 'Lightning Rod']
    }
    print(truncate_dict_strings(data, 20))
