import enum
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import regex as re
import weave
from pydantic import BaseModel, Field
from tenacity import after_log, before_sleep_log, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from wandbot.configs.chat_config import ChatConfig
from wandbot.models.llm import LLMModel
from wandbot.schema.api_status import APIStatus
from wandbot.utils import get_logger

logger = get_logger(__name__)
retry_chat_config = ChatConfig()

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
    "Reports, Experiments, Tables, Prompts, Launch, Weave and more",
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
    "Reports, Experiments, Tables, Prompts, Launch, Weave and more. Provide a link to the relevant "
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
        json_schema_extra={
            "description": "The reason to associate the intent with the query"
        }
    )
    label: Labels = Field(
        ..., 
        json_schema_extra={
            "description": "An intent associated with the query"
        }
    )


class Keyword(BaseModel):
    """A search phrase associated with the query"""

    keyword: str = Field(
        ...,
        json_schema_extra={
            "description": "A search phrase to get the most relevant information related to Weights & Biases from the web. "
            "This will be used to gather information required to answer the query"
        }
    )


class SubQuery(BaseModel):
    """A sub query that will help gather information required to answer the query"""

    query: str = Field(
        ...,
        json_schema_extra={
            "description": "The sub query that needs to be answered to answer the query. This will be used to define the "
            "steps required to answer the query"
        }
    )


class VectorSearchQuery(BaseModel):
    """A query for vector search"""

    query: str = Field(
        ...,
        json_schema_extra={
            "description": "A query to search for similar documents in the vector space. This will be used to find documents "
            "required to answer the query"
        }
    )


class EnhancedQuery(BaseModel):
    """An enhanced query"""

    language: str = Field(
        ...,
        json_schema_extra={
            "description": "The ISO code of language of the query"
        }
    )
    intents: List[Intent] = Field(
        ...,
        json_schema_extra={
            "description": f"A list of one or more intents associated with the query (minimum 1, maximum 5). Here are the possible intents that "
            f"can be associated with a query:\n{json.dumps(INTENT_DESCRIPTIONS)}"
        }
    )
    keywords: List[Keyword] = Field(
        ...,
        json_schema_extra={
            "description": "A list of diverse search terms associated with the query (minimum 1, maximum 5)."
        }
    )
    sub_queries: List[SubQuery] = Field(
        ...,
        json_schema_extra={
            "description": "A list of sub queries that break the query into smaller parts (minimum 1, maximum 5)"
        }
    )
    vector_search_queries: List[VectorSearchQuery] = Field(
        ...,
        json_schema_extra={
            "description": "A list of diverse queries to search for similar documents in the vector space (minimum 1, maximum 5)"
        }
    )

    standalone_query: str = Field(
        ...,
        json_schema_extra={
            "description": "A rephrased query that can be answered independently when chat history is available. If chat "
            "history is `None`, the original query must be copied verbatim"
        }
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
    "You are a weights & biases support manager tasked with enhancing support questions from users. "
    "You are given a conversation and a follow-up query. "
    "You goal to enhance the user query and render it using the tool provided."
    "\n\nChat History: \n\n"
    "{chat_history}"
)


def format_chat_history(chat_history: Optional[List[Tuple[str, str]]]) -> str:
    """Format chat history into a string"""
    if not chat_history:
        return "No chat history available."
    
    formatted = []
    for user_msg, assistant_msg in chat_history:
        formatted.extend([
            f"User: {user_msg}",
            f"Assistant: {assistant_msg}"
        ])
    return "\n".join(formatted)


class QueryEnhancer:
    def __init__(
        self,
        model_provider: str,
        model_name: str,
        temperature: float,
        fallback_model_provider: str,
        fallback_model_name: str,
        fallback_temperature: float,
        max_retries: int = 3
    ):
        self.model = LLMModel(
            provider=model_provider,
            model_name=model_name,
            temperature=temperature,
            response_model=EnhancedQuery,
            max_retries=max_retries
        )
        self.fallback_model = LLMModel(
            provider=fallback_model_provider,
            model_name=fallback_model_name,
            temperature=fallback_temperature,
            response_model=EnhancedQuery,
            max_retries=max_retries
        )

    @weave.op
    async def __call__(self, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhance a query with retries and fallback"""
        query = inputs.get("query", "")
        chat_history = inputs.get("chat_history")
        
        result = None
        llm_api_status = None
        
        try:
            # Try primary model
            result, llm_api_status = await self._try_enhance_query(self.model, query, chat_history)
            parsed_result = result.parse_output(query, chat_history)
        except Exception as e:
            logger.warning(f"Primary Query Enhancer model failed, trying fallback: {str(e)}")
            try:
                # Try fallback model
                result, llm_api_status = await self._try_enhance_query(self.fallback_model, query, chat_history)
                parsed_result = result.parse_output(query, chat_history)
            except Exception as e:
                logger.error(f"Both primary and fallback Query Enhancer models failed: {str(e)}")
                # If both models fail, raise the error
                raise Exception(f"Query enhancement failed: {str(e)}")
        
        # Only include LLM API status
        parsed_result["api_statuses"] = {
            "query_enhancer_llm_api": llm_api_status
        }
        return parsed_result

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(retry_chat_config.llm_max_retries),
        wait=wait_exponential(multiplier=retry_chat_config.llm_retry_multiplier,
                               min=retry_chat_config.llm_retry_min_wait, 
                               max=retry_chat_config.llm_retry_max_wait),
        before_sleep=lambda retry_state: (
            before_sleep_log(logger, log_level=logging.WARNING)(retry_state),
            logger.warning(
                f"Attempt {retry_state.attempt_number} failed. Retrying in {retry_state.next_action.sleep} seconds..."
            )
        )[1],
        reraise=True,
        after=after_log(logger, logging.ERROR)
    )
    @weave.op
    async def _try_enhance_query(
        self,
        model: LLMModel,
        query: str,
        chat_history: Optional[List[Tuple[str, str]]] = None
    ) -> Tuple[Dict[str, Any], APIStatus]:
        """Try to enhance a query using the given model"""
        messages = [
            {"role": "system", "content": ENHANCER_SYSTEM_PROMPT.format(
                chat_history=format_chat_history(chat_history)
            )},
            {"role": "user", "content": f"Question: {query}"},
            {"role": "user", "content": "!!! Tip: Make sure to answer in the correct format"}
        ]
        
        response, api_status = await model.create(messages=messages)
        if not api_status.success:
            raise Exception(api_status.error_info.error_message)
            
        return response, api_status
