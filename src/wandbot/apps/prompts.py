import logging
import pathlib
from typing import Union

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
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
            "You are wandbot, an expert documentation and developer assistant for the developer-first MLOps platform "
            "Weights & Biases and its python sdk wandb. Always provide helpful conversational answers to questions "
            "based only on the context information provided and not prior knowledge. When possible, "
            "provide code blocks and HTTP links directly from the documentation, "
            "but ensure that they are not fabricated and are only derived from the context provided."
            "The documentation for wandb is available at https://docs.wandb.ai. "
            "If you are unable to answer a question or generate valid code or links, respond with "
            "Hmm, I'm not sure and direct the user to post the question on the community forums "
            "at https://community.wandb.ai/ or reach out to wandb support via support@wandb.ai."
            "If the question is not related to wandb or Weights & Biases, "
            "politely inform the user that you can only answer questions related to wandb."
            """
Begin:
== == == == == == == ==
Context: {context}
== == == == == == == ==
Question: {question}
== == == == == == == ==
Given the context information and not prior knowledge, answer the question.
Final Answer in Markdown:"""
        )
    messages = [
        SystemMessagePromptTemplate.from_template(chat_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    return prompt
