
from loguru import logger
from snippets import add_callback2gen
from liteai.core import Message, ModelCard, ModelResponse, ToolDesc, Usage
from liteai.provider.open_ai import OpenAIProvider
from liteai.utils import acc_chunks, get_text_chunk, image2base64


class DeepSeekProvider(OpenAIProvider):
    key: str = "deepseek"
    allow_kwargs = {"do_sample", "stream", "temperature", "top_p", "max_tokens"}
    api_key_env = "DEEPSEEK_API_KEY"

    def __init__(self, api_key: str = None, base_url: str = None):
        self.base_url = base_url or "https://api.deepseek.com"
        super().__init__(api_key=api_key, base_url=self.base_url)



if __name__ == "__main__":
    provider = DeepSeekProvider()
    messages = [Message(role="user", content="你好")]
    from liteai.modelcard import get_modelcard
    model = get_modelcard("deepseek-chat")
    resp = provider.complete(messages=messages, model=model, stream=False)
    print(resp.content)
