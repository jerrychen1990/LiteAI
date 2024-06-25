import unittest

from unillm.core import Message
from unillm.api import completion
from unillm.utils import show_response


class TestZhipu(unittest.TestCase):

    def test_sync(self):
        system = "用英文回答我的问题, 80个单词以内"
        question = "介绍三首歌"
        model = "glm-4-air"
        messages = [Message(role="system", content=system),
                    Message(role="user", content=question)]
        response = completion(model=model, messages=messages, stream=False, temperature=0.)
        show_response(response)
        self.assertIsNotNone(response.usage)
        messages.extend([Message(role="assistant", content=response.choices[0].message.content),
                        Message(role="user", content="介绍第二首歌")])
        response = completion(model=model, messages=messages, stream=False, temperature=0.)
        show_response(response)
        self.assertIsNotNone(response.usage)
        self.assertEqual(len(response.choices), 1)
        self.assertTrue("Hotel California" in response.choices[0].message.content)
