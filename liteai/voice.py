#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/08/06 14:24:26
@Author  :   ChenHao
@Description  :   语音处理相关函数
@Contact :   jerrychen1990@gmail.com
'''
import io
import os
from typing import Iterable
from loguru import logger
from liteai.core import Voice
from snippets.utils import add_callback2gen
from pydub import AudioSegment

from pydub.playback import play


def build_voice(byte_stream: bytes | Iterable[bytes], file_path: str):
    if file_path:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        if isinstance(byte_stream, bytes):
            logger.debug(f"save voice to {file_path}")
            with open(file_path, "wb") as f:
                f.write(byte_stream)
        else:
            byte_stream = add_callback2gen(byte_stream, dump_voice_stream, path=file_path)
    return Voice(byte_stream=byte_stream, file_path=file_path)


def dump_voice_stream(voice_stream: Iterable[bytes], path: str):
    with open(path, "wb") as f:
        logger.debug(f"save voice to {path}")
        for chunk in voice_stream:
            if chunk is not None and chunk != b'\n':
                decoded_hex = chunk
                f.write(decoded_hex)


def get_duration(file_path: str):
    audio = AudioSegment.from_mp3(file_path)
    duration = len(audio) / 1000  # 时长以毫秒为单位
    return duration


def play_voice(voice: Voice):

    logger.debug(f"{type(voice.byte_stream)=}, {voice.file_path=}")

    if voice.file_path and os.path.exists(voice.file_path):
        logger.debug(f"playing voice from {voice.file_path}")

        with open(voice.file_path, "rb") as f:
            audio_bytes = f.read()

        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
        play(audio)
    else:
        logger.debug(f"playing voice from byte stream")
        import simpleaudio as sa
        audio_buffer = io.BytesIO()
        for chunk in voice.byte_stream:
            # 将每个块写入缓冲区
            audio_buffer.write(chunk)

            # 检查缓冲区大小，如果达到一定大小，则进行播放
            if audio_buffer.tell() > 4096:  # 例如，每4KB播放一次
                # 将缓冲区内容转换为 AudioSegment
                audio_buffer.seek(0)
                audio_segment = AudioSegment.from_file(audio_buffer, format="mp3")

                # 播放音频段
                play_obj = sa.play_buffer(audio_segment.raw_data,
                                          num_channels=audio_segment.channels,
                                          bytes_per_sample=audio_segment.sample_width,
                                          sample_rate=audio_segment.frame_rate)
                play_obj.wait_done()

                # 清空缓冲区
                audio_buffer = io.BytesIO()

        # 播放剩余部分
        if audio_buffer.tell() > 0:
            audio_buffer.seek(0)
            audio_segment = AudioSegment.from_file(audio_buffer, format="mp3")
            play_obj = sa.play_buffer(audio_segment.raw_data,
                                      num_channels=audio_segment.channels,
                                      bytes_per_sample=audio_segment.sample_width,
                                      sample_rate=audio_segment.frame_rate)
            play_obj.wait_done()
