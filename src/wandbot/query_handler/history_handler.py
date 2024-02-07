from _operator import itemgetter

from langchain_core.messages import get_buffer_string
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableBranch,
    Runnable,
)
from langchain_openai import ChatOpenAI

CONDENSE_PROMPT_SYSTEM_TEMPLATE = (
    "Given the following conversation between a user and an AI assistant and a follow up "
    "question from user, rephrase the follow up question to be a standalone question. Ensure "
    "that the standalone question summarizes the conversation and completes the follow up "
    "question with all the necessary context. If there is no history return the original question verbatim"
)


CONDENSE_PROMPT_MESSAGES = [
    ("system", CONDENSE_PROMPT_SYSTEM_TEMPLATE),
    (
        "human",
        """Rephrase the following question to be a standalone question based on the given history:

<history>
{chat_history}
</history>

<question>
{question}
</question>
""",
    ),
]


def load_standalone_query_chain(model: ChatOpenAI) -> Runnable:
    condense_prompt = ChatPromptTemplate.from_messages(CONDENSE_PROMPT_MESSAGES)

    condense_question_chain = (
        {
            "question": RunnablePassthrough(),
            "chat_history": itemgetter("chat_history")
            | RunnableLambda(get_buffer_string),
        }
        | condense_prompt
        | model
        | StrOutputParser()
    )

    return condense_question_chain


def load_query_condense_chain(
    model: ChatOpenAI,
) -> Runnable:
    standalone_query_chain = load_standalone_query_chain(
        model,
    )
    branch = RunnableBranch(
        (
            lambda x: True if x["chat_history"] else False,
            standalone_query_chain,
        ),
        (
            lambda x: False if x["chat_history"] else True,
            itemgetter("question"),
        ),
        itemgetter("question"),
    )

    return branch
