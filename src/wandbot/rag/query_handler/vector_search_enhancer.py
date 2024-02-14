from operator import itemgetter

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


class VectorSearchEnhancer:
    model: ChatModel = ChatModel()
    fallback_model: ChatModel = ChatModel(max_retries=6)

    def __init__(
        self,
        model: str = "gpt-4-0125-preview",
        fallback_model: str = "gpt-3.5-turbo-1106",
    ):
        self.model = model
        self.fallback_model = fallback_model
        self.prompt = ChatPromptTemplate.from_messages(
            QUERY_REWRITE_PROMPT_MESSAGES
        )
        self._chain = None

    @property
    def chain(self) -> Runnable:
        if self._chain is None:
            base_chain = self._load_chain(self.model)
            fallback_chain = self._load_chain(self.fallback_model)
            self._chain = base_chain.with_fallbacks([fallback_chain])
        return self._chain

    def _load_chain(self, model: ChatOpenAI) -> Runnable:
        query_rewrite_chain = create_structured_output_runnable(
            EnhancedQuery, model, self.prompt
        )
        question_rewrite_chain = query_rewrite_chain | RunnableLambda(
            lambda x: x.query_str
        )

        branch = RunnableBranch(
            (
                lambda x: x["avoid"],
                RunnableLambda(lambda x: []),
            ),
            (
                lambda x: not x["avoid"],
                question_rewrite_chain,
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
