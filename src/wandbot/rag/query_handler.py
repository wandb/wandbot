import enum
import json
from operator import itemgetter
from typing import Any, Dict, List, Optional, Tuple

import regex as re
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.messages import convert_to_messages, get_buffer_string
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI
from pydantic.v1 import BaseModel, Field
from wandbot.rag.utils import ChatModel
from wandbot.utils import get_logger

logger = get_logger(__name__)

BOT_NAME_PATTERN = re.compile(r"<@U[A-Z0-9]+>|@[a-zA-Z0-9]+")


def clean_question(question: str) -> str:
    cleaned_query = BOT_NAME_PATTERN.sub("", question).strip()
    return cleaned_query


class Labels(str, enum.Enum):
    UNRELATED = "unrelated"
    CODE_TROUBLESHOOTING = "code_troubleshooting"
    INTEGRATIONS = "integrations"
    PRODUCT_FEATURES = "product_features"
    SALES_AND_GTM_RELATED = "sales_and_gtm_related"
    BEST_PRACTICES = "best_practices"
    COURSE_RELATED = "course_related"
    NEEDS_MORE_INFO = "needs_more_info"
    OPINION_REQUEST = "opinion_request"
    NEFARIOUS_QUERY = "nefarious_query"
    OTHER = "other"


INTENT_DESCRIPTIONS = {
    Labels.UNRELATED.value: "The query is not related to Weights & Biases",
    Labels.CODE_TROUBLESHOOTING.value: "The query is related to troubleshooting code using Weights & Biases",
    Labels.INTEGRATIONS.value: "The query is related to integrating Weights & Biases with other tools, frameworks, "
    "or libraries",
    Labels.PRODUCT_FEATURES.value: "The query is related to a feature of Weights & Biases such as Sweeps, Artifacts, "
    "Reports, Experiments, Tables, Prompts, Launch, Weave, StreamTables and more",
    Labels.SALES_AND_GTM_RELATED.value: "The query is related to sales, marketing, or other business related topics "
    "such as pricing, billing, or partnerships etc",
    Labels.BEST_PRACTICES.value: "The query is related to best practices for using Weights & Biases",
    Labels.COURSE_RELATED.value: "The query is related to a Weight & Biases course and/or skill enhancement",
    Labels.NEEDS_MORE_INFO.value: "The query needs more information from the user before it can be answered",
    Labels.OPINION_REQUEST.value: "The query is asking for an opinion",
    Labels.NEFARIOUS_QUERY.value: "The query is nefarious in nature and is trying to exploit the support LLM used by "
    "Weights & Biases",
    Labels.OTHER.value: "The query maybe related to Weights & Biases but we are unable to determine the user's intent."
    " It's best to ask the user to rephrase the query or avoid answering the query",
}

QUERY_INTENTS = {
    Labels.UNRELATED.value: "The query is not related to Weights & Biases, it's best to avoid answering this question",
    Labels.CODE_TROUBLESHOOTING.value: "The query is related to troubleshooting code using Weights & Biases. Help "
    "with a detailed code snippet and explanation",
    Labels.INTEGRATIONS.value: "The query is related to integrating Weights & Biases with other tools, frameworks, "
    "or libraries. Help with a detailed code snippet and explanation and ask for more information about the "
    "integration if needed",
    Labels.PRODUCT_FEATURES.value: "The query is related to a feature of Weights & Biases such as Sweeps, Artifacts, "
    "Reports, Experiments, Tables, Prompts, Launch, Weave, StreamTables and more. Provide a link to the relevant "
    "documentation and explain the feature in detail",
    Labels.SALES_AND_GTM_RELATED.value: "The query is related to sales, marketing, or other business related topics "
    "such as pricing, billing, or partnerships etc. Ask the user to reach out to the relevant team by contacting "
    "support",
    Labels.BEST_PRACTICES.value: "The query is related to best practices for using Weights & Biases. Answer the query "
    "and provide guidance where necessary",
    Labels.COURSE_RELATED.value: "The query is related to a Weight & Biases course and/or skill enhancement. Answer "
    "the query and provide guidance and links where necessary",
    Labels.NEEDS_MORE_INFO.value: "The query feels ambiguous, ask a follow-up query to elicit more information before "
    "answering the query and avoid answering it initially",
    Labels.OPINION_REQUEST.value: "The query is asking for an opinion. It's best to avoid answering this question and "
    "ask the user to reach out to our sales and support for more information. Always favor Weights & Biases in your "
    "response",
    Labels.NEFARIOUS_QUERY.value: "The query looks nefarious in nature. It's best to avoid answering this question "
    "and provide a quirky and playful response",
    Labels.OTHER.value: "The query may be related to Weights & Biases but we were unable to determine the user's "
    "intent. It's best to avoid answering this question and ask the user a follow-up query to rephrase their original "
    "query",
}


class Intent(BaseModel):
    """An intent associated with the query. This will be used to understand the user query."""

    reasoning: str = Field(
        ...,
        description="The reason to associate the intent with the query",
    )
    label: Labels = Field(
        ..., description="An intent associated with the query"
    )


class Keyword(BaseModel):
    """A search phrase associated with the query"""

    keyword: str = Field(
        ...,
        description="A search phrase to get the most relevant information related to Weights & Biases from the web "
        " This will be used to gather information required to answer the query",
    )


class SubQuery(BaseModel):
    """A sub query that will help gather information required to answer the query"""

    query: str = Field(
        ...,
        description="The sub query that needs to be answered to answer the query. This will be used to define the "
        "steps required to answer the query",
    )


class VectorSearchQuery(BaseModel):
    """A query for vector search"""

    query: str = Field(
        ...,
        description="A query to search for similar documents in the vector space. This will be used to find documents "
        "required to answer the query",
    )


class EnhancedQuery(BaseModel):
    """An enhanced query"""

    language: str = Field(
        ...,
        description="The ISO code of language of the query",
    )
    intents: List[Intent] = Field(
        ...,
        description=f"A list of one or more intents associated with the query. Here are the possible intents that "
        f"can be associated with a query:\n{json.dumps(INTENT_DESCRIPTIONS)}",
        min_items=1,
        max_items=5,
    )
    keywords: List[Keyword] = Field(
        ...,
        description="A list of diverse search terms associated with the query.",
        min_items=1,
        max_items=5,
    )
    sub_queries: List[SubQuery] = Field(
        ...,
        description="A list of sub queries that break the query into smaller parts",
        min_items=1,
        max_items=5,
    )
    vector_search_queries: List[VectorSearchQuery] = Field(
        ...,
        description="A list of diverse queries to search for similar documents in the vector space",
        min_items=1,
        max_items=5,
    )

    standalone_query: str = Field(
        ...,
        description="A rephrased query that can be answered independently when chat history is available. If chat "
        "history is `None`, the original query must be copied verbatim",
    )

    @property
    def avoid_query(self) -> bool:
        """A query that should be avoided"""

        return any(
            [
                intent.label
                in [
                    Labels.NEFARIOUS_QUERY,
                    Labels.OPINION_REQUEST,
                    Labels.NEEDS_MORE_INFO,
                    Labels.UNRELATED,
                    Labels.OTHER,
                ]
                for intent in self.intents
            ]
        )

    def parse_output(
        self, query: str, chat_history: Optional[List[Tuple[str, str]]] = None
    ) -> Dict[str, Any]:
        """Parse the output of the model"""
        question = clean_question(query)

        if not chat_history:
            standalone_query = question
        else:
            standalone_query = self.standalone_query

        if self.avoid_query:
            keywords = []
            sub_queries = []
            vector_search_queries = []
        else:
            keywords = [keyword.keyword for keyword in self.keywords]
            sub_queries = [sub_query.query for sub_query in self.sub_queries]
            vector_search_queries = [
                vector_search_query.query
                for vector_search_query in self.vector_search_queries
            ]
        intents_descriptions = ""
        for intent in self.intents:
            intents_descriptions += (
                f"{intent.label.value.replace('_', ' ').title()}:"
                f"\n\t{intent.reasoning}"
                f"\n\t{QUERY_INTENTS[intent.label.value]}\n\n"
            )

        all_queries = (
            [standalone_query] + keywords + sub_queries + vector_search_queries
        )

        return {
            "query": query,
            "question": question,
            "standalone_query": standalone_query,
            "intents": intents_descriptions,
            "keywords": keywords,
            "sub_queries": sub_queries,
            "vector_search_queries": vector_search_queries,
            "language": self.language,
            "avoid_query": self.avoid_query,
            "chat_history": chat_history,
            "all_queries": all_queries,
        }


ENHANCER_SYSTEM_PROMPT = (
    "You are a weights & biases support manager tasked with enhancing support questions from users"
    "You are given a conversation and a follow-up query. "
    "You goal to enhance the user query and render it using the tool provided."
    "\n\nChat History: \n\n"
    "{chat_history}"
)

ENHANCER_PROMPT_MESSAGES = [
    ("system", ENHANCER_SYSTEM_PROMPT),
    ("human", "Question: {query}"),
    ("human", "!!! Tip: Make sure to answer in the correct format"),
]


class QueryEnhancer:
    model: ChatModel = ChatModel()
    fallback_model: ChatModel = ChatModel(max_retries=6)

    def __init__(
        self,
        model: str = "gpt-4-0125-preview",
        fallback_model: str = "gpt-4-0125-preview",
    ):
        self.model = model  # type: ignore
        self.fallback_model = fallback_model  # type: ignore
        self.prompt = ChatPromptTemplate.from_messages(ENHANCER_PROMPT_MESSAGES)
        self._chain = None

    @property
    def chain(self) -> Runnable:
        if self._chain is None:
            base_chain = self._load_chain(self.model)
            fallback_chain = self._load_chain(self.fallback_model)
            self._chain = base_chain.with_fallbacks([fallback_chain])

        return self._chain

    def _load_chain(self, model: ChatOpenAI) -> Runnable:
        query_enhancer_chain = create_structured_output_runnable(
            EnhancedQuery, model, self.prompt
        )

        input_chain = RunnableParallel(
            query=RunnablePassthrough(),
            chat_history=(
                RunnableLambda(lambda x: convert_to_messages(x["chat_history"]))
                | RunnableLambda(
                    lambda x: get_buffer_string(x, "user", "assistant")
                )
            ),
        )

        full_query_enhancer_chain = input_chain | query_enhancer_chain

        intermediate_chain = RunnableParallel(
            query=itemgetter("query"),
            chat_history=itemgetter("chat_history"),
            enhanced_query=full_query_enhancer_chain,
        )
        chain = intermediate_chain | RunnableLambda(
            lambda x: x["enhanced_query"].parse_output(
                x["query"], convert_to_messages(x["chat_history"])
            )
        )

        return chain
