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
from wandbot.rag.utils import ChatModel

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


class KeywordsEnhancer:
    model: ChatModel = ChatModel()
    fallback_model: ChatModel = ChatModel(max_retries=6)

    def __init__(
        self,
        model: str = "gpt-4-0125-preview",
        fallback_model: str = "gpt-3.5-turbo-1106",
    ):
        self.model = model  # type: ignore
        self.fallback_model = fallback_model  # type: ignore
        self.prompt = ChatPromptTemplate.from_messages(KEYWORDS_PROMPT_MESSAGES)
        self._chain = None

    @property
    def chain(self) -> Runnable:
        if self._chain is None:
            base_chain = self._load_chain(self.model)
            fallback_chain = self._load_chain(self.fallback_model)
            self._chain = base_chain.with_fallbacks([fallback_chain])

        return self._chain

    def _load_chain(self, model: ChatOpenAI) -> Runnable:
        keywords_extraction_chain = create_structured_output_runnable(
            KeywordsSchema, model, self.prompt
        )

        keywords_chain = keywords_extraction_chain | RunnableLambda(
            lambda x: [keyword.keyword for keyword in x.keywords]
        )

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

        chain = (
            RunnableParallel(
                question=itemgetter("standalone_question"),
                avoid=itemgetter("avoid_query"),
            )
            | branch
        )
        return chain
