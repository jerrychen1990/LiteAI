#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/08/29 14:20:40
@Author  :   ChenHao
@Description  :   讯飞适配器
@Contact :   jerrychen1990@gmail.com
'''
import datetime
import json
import os
import time
import _thread
import wave

from loguru import logger
import websocket


from liteai.core import ModelCard, Voice
from liteai.provider.base import BaseProvider
from wsgiref.handlers import format_date_time
from time import mktime
import websocket
import datetime
import hashlib
import base64
import hmac
from urllib.parse import urlencode
import ssl
from wsgiref.handlers import format_date_time
from datetime import datetime
from time import mktime
from websocket import WebSocket

from liteai.voice import file2voice, mp3to_wav
from snippets.logs import set_logger


STATUS_FIRST_FRAME = 0  # 第一帧的标识
STATUS_CONTINUE_FRAME = 1  # 中间帧标识
STATUS_LAST_FRAME = 2  # 最后一帧的标识


def create_url(api_key: str, api_secret: str):
    url = 'wss://ws-api.xfyun.cn/v2/iat'
    # 生成RFC1123格式的时间戳
    now = datetime.now()
    date = format_date_time(mktime(now.timetuple()))

    date = format_date_time(mktime(now.timetuple()))

    # 拼接字符串
    signature_origin = "host: " + "ws-api.xfyun.cn" + "\n"
    signature_origin += "date: " + date + "\n"
    signature_origin += "GET " + "/v2/iat " + "HTTP/1.1"
    # 进行hmac-sha256进行加密
    signature_sha = hmac.new(api_secret.encode('utf-8'), signature_origin.encode('utf-8'),
                             digestmod=hashlib.sha256).digest()
    signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')

    authorization_origin = "api_key=\"%s\", algorithm=\"%s\", headers=\"%s\", signature=\"%s\"" % (
        api_key, "hmac-sha256", "host date request-line", signature_sha)
    authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
    # 将请求的鉴权参数组合为字典
    v = {
        "authorization": authorization,
        "date": date,
        "host": "ws-api.xfyun.cn"
    }
    # 拼接鉴权参数，生成url
    url = url + '?' + urlencode(v)
    # print("date: ",date)
    # print("v: ",v)
    # 此处打印出建立连接时候的url,参考本demo的时候可取消上方打印的注释，比对相同参数时生成的url与自己代码生成的url是否一致
    # print('websocket url :', url)
    return url


def send_voice(ws: WebSocket,  voice: Voice, common_args: dict, business_args: dict, interval=0.1, frame_size=64*1000):
    status = STATUS_FIRST_FRAME  # 音频的状态信息，标识音频是第一帧，还是中间帧、最后一帧
    logger.debug(f"sending: {voice.file_path} to asr service")
    with wave.open(voice.file_path, "rb") as fp:
        while True:
            buf = fp.readframes(frame_size)
            if not buf:
                status = STATUS_LAST_FRAME
            # 第一帧处理
            # 发送第一帧音频，带business 参数
            # appid 必须带上，只需第一帧发送
            if status == STATUS_FIRST_FRAME:
                d = {"common": common_args,
                     "business": business_args,
                     "data": {"status": 0, "format": "audio/L16;rate=16000", "audio": str(base64.b64encode(buf), 'utf-8'), "encoding": "raw"}}
                status = STATUS_CONTINUE_FRAME
            # 中间帧处理
            elif status == STATUS_CONTINUE_FRAME:
                d = {"data": {"status": 1, "format": "audio/L16;rate=16000",
                              "audio": str(base64.b64encode(buf), 'utf-8'),
                              "encoding": "raw"}}
            # 最后一帧处理
            elif status == STATUS_LAST_FRAME:
                d = {"data": {"status": 2, "format": "audio/L16;rate=16000",
                              "audio": str(base64.b64encode(buf), 'utf-8'),
                              "encoding": "raw"}}
            # 模拟音频采样间隔
            # logger.debug(f"sending req:{jdumps({k:v for k,v in d.items() if k !='data'})} to ws")
            logger.debug(f"sending {len(buf)} bytes with {status=}")

            ws.send(json.dumps(d))
            time.sleep(interval)
            if status == STATUS_LAST_FRAME:
                time.sleep(1)
                break
    ws.close()

# 收到websocket消息的处理


def update_asr_text(text_dict: dict, message: str):
    try:
        logger.debug(f"receive msg:{message}")

        code = json.loads(message)["code"]
        sid = json.loads(message)["sid"]
        if code != 0:
            errMsg = json.loads(message)["message"]
            print("sid:%s call error:%s code is:%s" % (sid, errMsg, code))
        else:
            temp1 = json.loads(message)["data"]["result"]
            data = json.loads(message)["data"]["result"]["ws"]
            sn = temp1["sn"]
            if "rg" in temp1.keys():
                rep = temp1["rg"]
                rep_start = rep[0]
                rep_end = rep[1]
                for sn in range(rep_start, rep_end+1):
                    # print("before pop",whole_dict)
                    # print("sn",sn)
                    text_dict.pop(sn, None)
                    # print("after pop",whole_dict)
                results = ""
                for i in data:
                    for w in i["cw"]:
                        results += w["w"]
                text_dict[sn] = results
                # print("after add",whole_dict)
            else:
                results = ""
                for i in data:
                    for w in i["cw"]:
                        results += w["w"]
                text_dict[sn] = results
            # print("sid:%s call success!,data is:%s" % (sid, json.dumps(data, ensure_ascii=False)))
    except Exception as e:
        print("receive msg,but parse exception:", e)


# 收到websocket错误的处理
def on_error(ws, error):
    logger.error("### error:", error)


# 收到websocket关闭的处理
def on_close(ws, a, b):
    logger.info("web socket closed")


DEFAULT_BUSINESS_ARGS = {
    "domain": "iat",
    "language": "zh_cn",
    "accent": "mandarin",
    "vad_eos": 10000,
    "dwa": "wags"
}


class XunfeiProvider(BaseProvider):
    key: str = "xunfei"
    allow_kwargs = {"stream", "temperature", "top_p", "max_tokens", "meta"}
    api_key_env = "XUNFEI_API_KEY"

    def __init__(self, api_key: str = None, api_secret: str = None, ** kwargs):
        super().__init__(api_key=api_key)
        self.api_secret = api_secret or os.environ.get("XUNFEI_API_SECRET")
        logger.debug(f"xunfei provider init finished")

    def asr(self, voice: Voice, model: ModelCard, app_id: str,  business_args: dict = DEFAULT_BUSINESS_ARGS, interval=0.1, frame_size=64*1000, ** kwargs) -> str:
        if voice.file_path.endswith(".mp3"):
            file_path = mp3to_wav(mp3_path=voice.file_path)
            voice = file2voice(file_path=file_path)

        websocket.enableTrace(False)
        ws_url = create_url(self.api_key, self.api_secret)
        logger.debug(f"asr ws_url: {ws_url}")
        text_dict = dict()

        def onopen(ws: WebSocket):
            _thread.start_new_thread(send_voice, (ws, voice,  {"app_id": app_id}, business_args, interval, frame_size))

        def on_message(ws: WebSocket, message):
            logger.debug(f"get message:{message}")
            update_asr_text(text_dict, message)

        ws = websocket.WebSocketApp(ws_url, on_message=on_message, on_error=on_error, on_close=on_close, on_open=onopen)
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
        whole_words = ""
        logger.debug(f"{text_dict=}")
        for i in sorted(text_dict.keys()):
            whole_words += text_dict[i]
        return whole_words


if __name__ == "__main__":
    # load_dotenv()
    set_logger("dev", __name__)
    # print(os.environ.get(XunfeiProvider.api_key_env))
    xunfei_provider = XunfeiProvider()
    file_path = mp3to_wav(mp3_path="./data/hello.mp3")
    voice = file2voice(file_path=file_path)
    voice_text = xunfei_provider.asr(voice=voice, model=ModelCard(name="xunfei_asr", description="讯飞语音识别", provider="xunfei"), app_id="9b5cc933")
    print(voice_text)
