import json
import logging
import pathlib
from typing import Union

from llama_index import ChatPromptTemplate
from llama_index.llms import ChatMessage, MessageRole

logger = logging.getLogger(__name__)


def load_chat_prompt(f_name: Union[pathlib.Path, str] = None) -> ChatPromptTemplate:
    f_name = pathlib.Path(f_name)
    template = json.load(f_name.open("r"))
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=template["system_template"]),
        ChatMessage(role=MessageRole.USER, content=template["human_template"]),
    ]

    prompt = ChatPromptTemplate(messages)
    return prompt
