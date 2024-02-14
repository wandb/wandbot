from _operator import itemgetter
from langchain_core.messages import convert_to_messages, get_buffer_string
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI

from wandbot.rag.utils import ChatModel

CONDENSE_PROMPT_SYSTEM_TEMPLATE = """Given the following conversation and a follow up question, rephrase the follow up \
question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone Question:"""


CONDENSE_PROMPT_MESSAGES = [
    (
        "system",
        CONDENSE_PROMPT_SYSTEM_TEMPLATE,
    ),
]


class CondenseQuestion:
    model: ChatModel = ChatModel()
    fallback_model: ChatModel = ChatModel(max_retries=6)

    def __init__(
        self,
        model: str = "gpt-4-1106-preview",
        fallback_model="gpt-4-1106-preview",
    ):
        self.model = model
        self.fallback_model = fallback_model
        self.prompt = ChatPromptTemplate.from_messages(CONDENSE_PROMPT_MESSAGES)
        self._chain = None

    @property
    def chain(self) -> Runnable:
        if self._chain is None:
            base_chain = self._load_chain(self.model)
            fallback_chain = self._load_chain(self.fallback_model)
            self._chain = base_chain.with_fallbacks([fallback_chain])

        return self._chain

    def _load_chain(self, model: ChatOpenAI) -> Runnable:
        base_chain = (
            RunnableParallel(
                question=RunnablePassthrough(),
                chat_history=(
                    RunnableLambda(
                        lambda x: convert_to_messages(x["chat_history"])
                    )
                    | RunnableLambda(
                        lambda x: get_buffer_string(x, "user", "assistant")
                    )
                ),
            )
            | self.prompt
            | model
            | StrOutputParser()
        )

        chain = RunnableBranch(
            (
                lambda x: True if x["chat_history"] else False,
                base_chain,
            ),
            (
                lambda x: False if x["chat_history"] else True,
                itemgetter("question"),
            ),
            itemgetter("question"),
        )
        return chain
