from operator import itemgetter

from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableSerializable,
    RunnableLambda,
    RunnableBranch,
    RunnableParallel,
    Runnable,
)
from langchain_openai import ChatOpenAI
from pydantic.v1 import BaseModel, Field

QUERY_REWRITE_SYSTEM_PROMPT = (
    "You are a Weights & Biases support manager. "
    "Your goal is to enhance the user query by rewriting it for similarity search. "
    "Rewrite the given query into a clear, specific, and formal request for retrieving relevant information from a vector database"
)

QUERY_REWRITE_PROMPT_MESSAGES = [
    ("system", QUERY_REWRITE_SYSTEM_PROMPT),
    ("human", "Enhance the following query i.:\n\n{question}"),
    ("human", "Tip: Make sure to answer in the correct format"),
]


class EnhancedQuery(BaseModel):
    "A query suitable for similarity search in a vectorstore"
    query_str: str = Field(
        ..., description="A query suitable for similarity search and retrieval"
    )


def load_query_rewrite_chain(model: ChatOpenAI) -> RunnableSerializable:

    query_rewrite_prompt = ChatPromptTemplate.from_messages(
        QUERY_REWRITE_PROMPT_MESSAGES
    )

    query_rewrite_chain = create_structured_output_runnable(
        EnhancedQuery, model, query_rewrite_prompt
    )

    question_rewrite_chain = query_rewrite_chain | RunnableLambda(
        lambda x: x.query_str
    )

    return question_rewrite_chain


def load_vectorsearch_enhancement_chain(model: ChatOpenAI) -> Runnable:
    vectorsearch_chain = load_query_rewrite_chain(model)

    branch = RunnableBranch(
        (
            lambda x: x["avoid"],
            RunnableLambda(lambda x: []),
        ),
        (
            lambda x: not x["avoid"],
            vectorsearch_chain,
        ),
        RunnableLambda(lambda x: []),
    )

    return (
        RunnableParallel(
            question=itemgetter("standalone_question"),
            avoid=itemgetter("avoid_query"),
        )
        | branch
    )
