import logging
import pathlib
from typing import Union

from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)

logger = logging.getLogger(__name__)


def load_hyde_prompt(f_name: Union[pathlib.Path, str] = None) -> ChatPromptTemplate:
    if isinstance(f_name, str) and f_name:
        f_name = pathlib.Path(f_name)
    if f_name and f_name.is_file():
        hyde_template = f_name.open("r").read()
    else:
        logger.warning(
            f"No hyde_prompt provided. Using default hyde prompt from {__name__}"
        )
        hyde_template = (
            """Please answer the user's question about the Weights & Biases, W&B or wandb. """
            """Provide a detailed code example with explanations whenever possible. """
            """If the question is not related to Weights & Biases or wandb """
            """just say "I'm not sure how to respond to that."\n"""
            """\n"""
            """Begin\n"""
            """==========\n"""
            """Question: {question}\n"""
            """Answer:"""
        )
    messages = [
        SystemMessagePromptTemplate.from_template(hyde_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    hyde_prompt = ChatPromptTemplate.from_messages(messages)
    return hyde_prompt


def load_chat_prompt(f_name: Union[pathlib.Path, str] = None) -> ChatPromptTemplate:
    if isinstance(f_name, str) and f_name:
        f_name = pathlib.Path(f_name)
    if f_name and f_name.is_file():
        chat_template = f_name.open("r").read()
    else:
        logger.warning(
            f"No chat prompt provided. Using default chat prompt from {__name__}"
        )
        chat_template = (
            "Your task is to serve as an AI assistant for the open source library wandb. You should answer questions "
            "based on the given extracted parts of a long document and the question. You should provide a "
            "conversational answer with a hyperlink to the documentation only if it is explicitly listed as a source "
            "in the context. When possible, provide code blocks and HTTP links directly from the documentation, but "
            "ensure that they are not fabricated. If you cannot answer the question or generate valid code or links, "
            "simply respond with 'Hmm, I'm not sure.' If the question is not related to wandb or Weights & Biases, "
            "politely inform the user that you can only answer questions related to wandb. The documentation for wandb "
            "is available at https://docs.wandb.ai."
            """
Begin:
== == == == == == == ==
Context: {context}
== == == == == == == ==
Question: {question}
== == == == == == == ==
Final Answer in Markdown:"""
        )
    messages = [
        SystemMessagePromptTemplate.from_template(chat_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    return prompt
