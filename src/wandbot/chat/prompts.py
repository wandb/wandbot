"""This module provides functionality for loading chat prompts.

The main function in this module is `load_chat_prompt`, which loads a chat prompt from a given JSON file.
The JSON file should contain two keys: "system_template" and "human_template", which correspond to the system and user messages respectively.

Typical usage example:

  from wandbot.chat import prompts

  chat_prompt = prompts.load_chat_prompt('path_to_your_json_file.json')
"""

import json
import logging
import pathlib
from typing import Union

from llama_index import ChatPromptTemplate
from llama_index.llms import ChatMessage, MessageRole

logger = logging.getLogger(__name__)


from string import Formatter


def partial_format(s, **kwargs):
    place_holders = set(
        [x[1] for x in Formatter().parse(s) if x[1] is not None]
    )
    replacements = {k: kwargs.get(k, "{" + k + "}") for k in place_holders}
    return s.format(**replacements)


def load_chat_prompt(
    f_name: Union[pathlib.Path, str] = None,
    system_template: Union[pathlib.Path, str] = None,
    language_code: str = "en",
    query_intent: str = "",
) -> ChatPromptTemplate:
    """
    Loads a chat prompt from a given file.

    This function reads a JSON file specified by f_name and constructs a ChatPromptTemplate
    object from the data. The JSON file should contain two keys: "system_template" and "human_template",
    which correspond to the system and user messages respectively.

    Args:
        f_name: A string or a pathlib.Path object representing the path to the JSON file.
            If None, a default path is used.

    Returns:
        A ChatPromptTemplate object constructed from the data in the JSON file.
    """
    f_name = pathlib.Path(f_name)

    template = json.load(f_name.open("r"))
    human_template = partial_format(
        template["human_template"],
        language_code=language_code,
        query_intent=query_intent,
    )
    messages = [
        ChatMessage(
            role=MessageRole.SYSTEM, content=open(system_template).read()
        ),
        ChatMessage(
            role=MessageRole.USER, content=human_template
        ),  # template["human_template"]),
    ]

    prompt = ChatPromptTemplate(messages)
    return prompt
