#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/09/03 16:45:23
@Author  :   ChenHao
@Description  :   命令行工具
@Contact :   jerrychen1990@gmail.com
'''


import os
import time

import click
from loguru import logger
from liteai.api import chat
from liteai.core import Message
from liteai.utils import set_logger
from snippets.decorators import multi_thread
from snippets import dump, load
from snippets.logs import ChangeLogLevelContext


@click.group(name="liteai")
def cli():
    pass


@click.command("batch")
@click.option("--model", "-m", help="模型名称")
@click.option("--data_path", "-d", help="输入文件路径，支持.xlsx, .csv, .json")
@click.option("--work_num", "-w", default=4, help="并发数")
@click.option("--input_column", default="input", help="输入列名")
@click.option("--image_column", help="图片列名")
@click.option("--image_dir", help="图片目录, 输入图片列名时必须存在")
@click.option("--overwrite", default=False, help="是否覆盖原文件, 未填写output_path时生效")
@click.option("--system", help="system message,", default=None)
@click.option("--temperature", default=0.7, help="温度")
@click.option("--top_p", default=0.7, help="top_p")
@click.option("--log_level", default="INFO", help="日志级别")
@click.option("--output_path", help="输出文件路径", default=None)
def batch(data_path: str, model: str, work_num=4, input_column="input", image_column: str = None, image_dir=None,
          overwrite: True = False, output_path: str = None, system: str = None, temperature=0.7, top_p=.7, log_level="INFO"):
    """
    批量处理数据
    """
    st = time.time()
    data = load(data_path)
    logger.info(f"loaded {len(data)} records from {data_path}")
    if image_column:
        assert image_dir, "image_dir is required when image_column is set"
        if not os.path.exists(image_dir):
            raise ValueError(f"image_dir {image_dir} not exists")

    def _func(item):
        image_path = os.path.join(image_dir, item[image_column]) if image_column else None
        messages = [Message(role="user", content=item[input_column], image=image_path)]
        if system:
            messages.insert(0, Message(role="system", content=system))
        # logger.info(f"{log_level=}")

        resp = chat(model=model, messages=messages, stream=False, top_p=top_p, temperature=temperature)
        item[f"{model}_response"] = resp.content

    with ChangeLogLevelContext(module_name="liteai", sink_type="stdout", level=log_level.upper()):
        batch_func = multi_thread(work_num=work_num, return_list=True)(_func)
        batch_func(data=data)
        if not output_path:
            base, ext = os.path.splitext(data_path)

            if overwrite:
                output_path = data_path
            else:
                output_path = base + f"_{model}" + ext
    logger.info(f"processed {len(data)} records in {time.time()-st:.2f} seconds, dump to {output_path}")
    dump(data, output_path)


cli.add_command(batch)


if __name__ == "__main__":
    set_logger(__name__)
    cli()
