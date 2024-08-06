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
import os
import sys
from loguru import logger
from PIL import Image
from liteai.core import ModelResponse
from snippets import batchify


def show_response(response: ModelResponse, batch_size=10):
    content = response.content
    if isinstance(content, str):
        logger.info(content)
    else:
        for item in batchify(content, batch_size):
            logger.info("".join(item))


def set_logger(module):
    print(f"setting logger for {module=}")
    if 0 in logger._core.handlers:
        logger.remove(0)
    UNILLM_ENV = os.environ.get("UNILLM_ENV", "DEV")

    dev_fmt = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> [<level>{level: <8}</level>] - <cyan>{file}</cyan>:<cyan>{line}</cyan>[<cyan>{name}</cyan>:<cyan>{function}</cyan>] - <level>{message}</level>"
    if UNILLM_ENV.upper() == "DEV":
        logger.add(sys.stdout, level="DEBUG", filter=lambda r: module in r["name"], format=dev_fmt, enqueue=True, colorize=True)


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


def get_chunk_data(chunk):
    choices = chunk.choices
    if choices:
        choice = choices[0]
        # logger.debug(f"{choice=}")
        if choice.delta and choice.delta.content:
            delta_content = choice.delta.content
            # logger.info(f"{delta_content}")
            return delta_content


def acc_chunks(acc):
    resp_msg = "".join(acc).strip()
    logger.debug(f"model generate answer:{resp_msg}")


if __name__ == "__main__":
    data = {
        'name': 'Pikachu',
        'description': 'Pikachu is an Electric-type Pokémon introduced in Generation I.',
        'abilities': ['Static', 'Lightning Rod']
    }
    print(truncate_dict_strings(data, 20))
