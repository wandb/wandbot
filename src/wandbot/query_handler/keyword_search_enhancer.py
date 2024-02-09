from operator import itemgetter
from typing import List

from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
)
from langchain_openai import ChatOpenAI
from pydantic.v1 import BaseModel, Field

KEYWORDS_SYSTEM_PROMPT = (
    "You are a Weights & Biases support manager. "
    "Your goal is to enhance the user query by adding a list of keywords used for web search."
)

KEYWORDS_PROMPT_MESSAGES = [
    ("system", KEYWORDS_SYSTEM_PROMPT),
    (
        "human",
        "Enhance the following query related to weights and biases for web search.:\n\n{question}",
    ),
    ("human", "Tip: Make sure to answer in the correct format"),
]


class Keyword(BaseModel):
    """A Keyword to search for on the wen"""

    keyword: str = Field(
        ...,
        description="A search term for getting the most relevant information required to answer the query",
    )


class KeywordsSchema(BaseModel):
    "A list of search keywords to enhance the search query"
    keywords: List[Keyword] = Field(
        ...,
        description="List of five different search terms",
        min_items=0,
        max_items=5,
    )


def load_keywords_extraction_chain(model: ChatOpenAI) -> Runnable:
    keywords_prompt = ChatPromptTemplate.from_messages(KEYWORDS_PROMPT_MESSAGES)

    keywords_extraction_chain = create_structured_output_runnable(
        KeywordsSchema, model, keywords_prompt
    )

    keywords_chain = keywords_extraction_chain | RunnableLambda(
        lambda x: [keyword.keyword for keyword in x.keywords]
    )

    return keywords_chain


def load_keywords_enhancement_chain(model: ChatOpenAI) -> Runnable:
    keywords_chain = load_keywords_extraction_chain(model)

    branch = RunnableBranch(
        (
            lambda x: x["avoid"],
            RunnableLambda(lambda x: []),
        ),
        (
            lambda x: not x["avoid"],
            keywords_chain,
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
