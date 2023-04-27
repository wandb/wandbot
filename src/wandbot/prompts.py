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


def load_eval_prompt(f_name: Union[pathlib.Path, str] = None) -> ChatPromptTemplate:
    if isinstance(f_name, str) and f_name:
        f_name = pathlib.Path(f_name)
    if f_name and f_name.is_file():
        human_template = f_name.open("r").read()
    else:
        logger.warning(
            f"No human prompt provided. Using default human prompt from {__name__}"
        )

        human_template = """You are an evaluator for the W&B chatbot.
        You are given a question, the chatbot's answer, and the original answer, and are asked to score the chatbot's answer as either CORRECT or INCORRECT.
        Note that sometimes, the original answer is not the best answer, and sometimes the chatbot's answer is not the best answer. 
        You are evaluating the chatbot's answer only. 
        Example Format:
        QUESTION: question here
        CHATBOT ANSWER: student's answer here
        ORIGINAL ANSWER: original answer here
        GRADE: CORRECT or INCORRECT here
        Please remember to grade them based on being factually accurate. Begin!
        QUESTION: {query}
        CHATBOT ANSWER: {result}
        ORIGINAL ANSWER: {answer}
        GRADE:"""

    system_message_prompt = SystemMessagePromptTemplate.from_template(
        "You are a helpful assistant"
    )
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    return chat_prompt
