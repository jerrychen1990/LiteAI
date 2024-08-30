#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/08/06 14:24:26
@Author  :   ChenHao
@Description  :   语音处理相关函数
@Contact :   jerrychen1990@gmail.com
'''
import io
import itertools
import os
import threading
import time
from typing import Iterable
from loguru import logger
from liteai.config import DEFAULT_VOICE_CHUNK_SIZE, MAX_PLAY_SECONDS
from liteai.core import Voice
from snippets.utils import add_callback2gen
from pydub import AudioSegment

from pydub.playback import play


def build_voice(byte_stream: bytes | Iterable[bytes], file_path: str = None, overwrite=False):
    if file_path:
        if os.path.exists(file_path) and not overwrite:
            pass
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if isinstance(byte_stream, bytes):
            logger.debug(f"save voice to {file_path}")
            with open(file_path, "wb") as f:
                f.write(byte_stream)
        else:
            byte_stream = add_callback2gen(byte_stream, dump_voice_stream, path=file_path)
    return Voice(byte_stream=byte_stream, file_path=file_path)


def file2voice(file_path: str, chunk_size=None):
    def byte_gen():
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    return
                yield chunk
                logger.debug(f"yielding chunk, with size {len(chunk)}")
    byte_stream = byte_gen()
    return build_voice(byte_stream=byte_stream, file_path=file_path, overwrite=False)


def mp3to_wav(mp3_path: str, wav_path: str = None) -> str:
    audio: AudioSegment = AudioSegment.from_mp3(mp3_path)
    if not wav_path:
        wav_path = mp3_path.replace(".mp3", ".wav")
    audio.export(wav_path, format="wav")
    return wav_path


def dump_voice_stream(voice_stream: Iterable[bytes] | bytes, path: str):
    with open(path, "wb") as f:
        logger.debug(f"save voice to {path}")
        if isinstance(voice_stream, bytes):
            f.write(voice_stream)
        else:
            for chunk in voice_stream:
                if chunk is not None and chunk != b'\n':
                    decoded_hex = chunk
                    f.write(decoded_hex)


def get_duration(file_path: str):
    audio = AudioSegment.from_mp3(file_path)
    duration = len(audio) / 1000  # 时长以秒为单位
    return duration


def play_file(file_path: str, max_seconds: int = None):
    logger.debug(f"playing voice from {file_path}")
    with open(file_path, "rb") as f:
        audio_bytes = f.read()
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
    if max_seconds:
        audio = audio[:max_seconds * 1000]
    play(audio)


def play_voice(voice: Voice, buffer_size=DEFAULT_VOICE_CHUNK_SIZE, max_seconds=None):

    logger.debug(f"{type(voice.byte_stream)=}, {voice.file_path=}")

    if voice.file_path and os.path.exists(voice.file_path):
        play_file(voice.file_path, max_seconds=max_seconds)
    else:
        logger.debug(f"playing voice from byte stream")
        voice.byte_stream, play_stream = itertools.tee(voice.byte_stream)
        play_bytes(play_stream, buffer_size, max_seconds=max_seconds)


def play_bytes(byte_stream: Iterable[bytes], min_buffer_size=8192*10, max_seconds=None):
    audio_buffer = io.BytesIO()
    producer_finished = threading.Event()
    offset = 0

    def _buf_reader():
        for chunk in byte_stream:
            # logger.debug(f"write {len(chunk)} bytes to buffer")
            audio_buffer.write(chunk)
            audio_buffer.flush()
        producer_finished.set()

    thread = threading.Thread(target=_buf_reader)
    thread.start()
    seconds_left = max_seconds if max_seconds else MAX_PLAY_SECONDS

    def play_segment(check_min_size: bool, seconds_left: int):
        end_offset = audio_buffer.tell()
        raw_size = end_offset - offset
        # logger.debug(f"{producer_finished.is_set()=}, {raw_size=}, {offset=}, {end_offset=}")
        if raw_size > 0:
            # logger.debug(f"{producer_finished.is_set()=}, {raw_size=}, {offset=}, {end_offset=}")
            if not check_min_size or raw_size >= min_buffer_size:
                audio_buffer.seek(offset)
                audio_segment = AudioSegment.from_file(audio_buffer, format="mp3")
                audio_segment = audio_segment[:seconds_left * 1000]
                seconds_left -= len(audio_segment) / 1000
                duration = len(audio_segment)
                logger.debug(f"play audio segment with size:{len(audio_segment.raw_data)}, {raw_size=}, duration:{duration}")
                play(audio_segment)
                return end_offset, seconds_left
        else:
            time.sleep(0.2)
        return offset, seconds_left

    while True:
        if producer_finished.is_set():
            offset, seconds_left = play_segment(False, seconds_left)
            break
        else:
            offset, seconds_left = play_segment(True, seconds_left)
        if seconds_left <= 0:
            break

    # thread.join()
