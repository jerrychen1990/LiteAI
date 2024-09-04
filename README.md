# LiteAI

提供比[LiteLLM](https://github.com/BerriAI/litellm)更简洁、范围更广的API，并支持更多的AI模型。

- 增加国内厂商的sdk接入
- 增加embedding/agent等接口
- 简化模型的输入输出信息，方便下游处理

| 厂商                                                                                     | chat | 流式chat | Image输入 | Image输出 | Embedding | ToolUse | TTS | ASR |
| ---------------------------------------------------------------------------------------- | ---- | -------- | --------- | --------- | --------- | ------- | --- | --- |
| [ZhipuAI](https://open.bigmodel.cn/dev/api#glm-4)                                           | ✅   | ✅       | ✅        |           | ✅        | ✅      |     |     |
| [Qwen](https://help.aliyun.com/zh/dashscope/qwen-api-details)                               | ✅   | ✅       | ✅        |           |           |         |     |     |
| [OpenAI](https://platform.openai.com/docs/guides/chat-completions)                          | ✅   | ✅       |           |           |           |         |     |     |
| [Doubao](https://www.volcengine.com/docs/82379/1263482)                                     | ✅   | ✅       |           |           |           |         |     |     |
| [MiniMax](https://platform.minimaxi.com/document/Announcement?key=66701c5e1d57f38758d58180) | ✅   | ✅       |           |           |           |         | ✅  |     |
| [Xunfei](https://www.xfyun.cn/doc/asr/voicedictation/API.html)                              |      |          |           |           |           |         |     | ✅  |
| [Ollama](https://ollama.com/)                                                               | ✅   | ✅       |           |           |           |         |     |     |
| [SiliconFlow](https://docs.siliconflow.cn/reference)                                                               | ✅   | ✅       |           |           |           |         |     |     |



## Install
```bash
pip install -U liteai
```

## 批量请求模型
```bash
liteai batch \
-d liteai/data/batch_test.jsonl \
-m glm-4v \
-w 2 \
--input_column text \
--image_column image \
--image_dir liteai/data \
--system "用英文回答" \
--output_path output/bath_test_glm4v.xlsx
```
具体参数定义使用
```bash liteai batch --help```查看