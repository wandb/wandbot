import enum
import json
import os
import re
from typing import List, Optional

import cohere
import instructor
import openai
import tiktoken
from llama_index.llms import ChatMessage, MessageRole
from llama_index.llms.generic_utils import messages_to_history_str
from openai import OpenAI
from pydantic import BaseModel, Field
from pydantic.v1 import BaseModel as BaseModelV1
from pydantic_settings import BaseSettings, SettingsConfigDict
from tenacity import retry, stop_after_attempt, wait_random_exponential
from wandbot.chat.schemas import ChatRequest
from wandbot.database.schemas import QuestionAnswer
from wandbot.utils import get_logger

logger = get_logger(__name__)


CONDENSE_PROMPT_SYSTEM_TEMPLATE = (
    "Given the following conversation between a user and an AI assistant and a follow up "
    "question from user, rephrase the follow up question to be a standalone question. Ensure "
    "that the standalone question summarizes the conversation and completes the follow up "
    "question with all the necessary context."
    + """
Chat History:
  {chat_history}
"""
)
CONDENSE_PROMPT_USER_TEMPLATE = """Follow Up Input: {question}
  Standalone question:"""


class StandaloneQuestion(BaseModel):
    question: str = Field(
        ...,
        description="A standalone question that summarizes the conversation and completes the follow up question with "
        "all the necessary context",
    )


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


class MultiLabel(BaseModel):
    label: Labels = Field(..., description="The label for the query")
    reasoning: str = Field(
        ...,
        description="The reason for the assigning the label to the query",
    )


class SubQuery(BaseModel):
    """Correctly resolved subquery from the given query"""

    query: str = Field(
        ..., description="Informative sub-query from the given user query. "
    )


class Keyword(BaseModel):
    """Keyword extracted from the given query to search for relevant articles"""

    keyword: str = Field(
        ...,
        description="A Keyword that can be used to retrieve for relevant articles required to answer the query",
    )


class EnhancedQuery(BaseModel):
    """An enhanced query with keywords, intent and sub-queries related to the query"""

    predicted_labels: List[MultiLabel] = Field(
        ..., description="The predicted labels for the query"
    )
    keywords: List[Keyword] = Field(
        ...,
        description="List of keywords and key phrases to search for",
        min_items=0,
        max_items=5,
    )
    sub_queries: List[SubQuery] = Field(
        ...,
        description="List of sub-queries that need be answered to resolve the query",
        min_items=0,
        max_items=3,
    )


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
    "answering the query",
    Labels.OPINION_REQUEST.value: "The query is asking for an opinion. It's best to avoid answering this question and "
    "ask the user to reach out to the relevant team by contacting support for more information",
    Labels.NEFARIOUS_QUERY.value: "The query looks nefarious in nature. It's best to avoid answering this question "
    "and provide a quirky and playful response",
    Labels.OTHER.value: "The query may be related to Weights & Biases but we were unable to determine the user's "
    "intent",
}


class ResolvedQuery(BaseModelV1):
    cleaned_query: str | None
    query: str | None
    condensed_query: str | None
    intent: str | None
    language: str | None
    chat_history: List[ChatMessage] | None = None


def get_chat_history(
    chat_history: List[QuestionAnswer] | None,
) -> Optional[List[ChatMessage]]:
    """Generates a list of chat messages from a given chat history.

    This function takes a list of QuestionAnswer objects and transforms them into a list of ChatMessage objects. Each
    QuestionAnswer object is split into two ChatMessage objects: one for the user's question and one for the
    assistant's answer. If the chat history is empty or None, the function returns None.

    Args: chat_history: A list of QuestionAnswer objects representing the history of a chat. Each QuestionAnswer
    object contains a question from the user and an answer from the assistant.

    Returns: A list of ChatMessage objects representing the chat history. Each ChatMessage object has a role (either
    'USER' or 'ASSISTANT') and content (the question or answer text). If the chat history is empty or None,
    the function returns None.
    """
    if not chat_history:
        return None
    else:
        messages = [
            [
                ChatMessage(
                    role=MessageRole.USER, content=question_answer.question
                ),
                ChatMessage(
                    role=MessageRole.ASSISTANT, content=question_answer.answer
                ),
            ]
            for question_answer in chat_history
        ]
        return [item for sublist in messages for item in sublist]


class OpenaiQueryEnhancer:
    def __init__(
        self, client: openai.OpenAI, model: str = "gpt-4-1106-preview"
    ):
        self.client = instructor.patch(client)
        self.model = model

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
    )
    def enhance_query(self, resolved_query: ResolvedQuery) -> EnhancedQuery:
        query_object = dict(
            query=resolved_query.condensed_query,
            intent_hints=resolved_query.intent,
            language=resolved_query.language,
        )
        query_object_str = json.dumps(query_object, indent=2)

        return self.client.chat.completions.create(
            model=self.model,
            response_model=EnhancedQuery,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Weights & Biases support manager. Your goal is to enhance the user query by adding "
                        "the following information to the query:\n"
                        "1.  Tag and classify query. Here are the descriptions of the available labels\n"
                        + "\n".join(
                            [
                                f"{label}: {description}"
                                for label, description in INTENT_DESCRIPTIONS.items()
                            ]
                        )
                        + "\n"
                        "2.  A list of keywords and key phrases to search for relevant articles to answer the query.\n"
                        "3.  A list of sub-queries that need be answered to answer the query\n\n"
                        "**Note**: Do not fill in the keywords and sub-queries if the query is nefarious, "
                        "opinion related or unrelated to Weights & Biases\n"
                    ),
                },
                {
                    "role": "user",
                    "content": f"Enhance the following user query related to Weights & Biases:\n{query_object_str}",
                },
            ],
        )  # type: ignore

    def condense_question(
        self, chat_history: List[ChatMessage], latest_message: str
    ) -> StandaloneQuestion:
        """Condense a conversation history and latest user message to a standalone question."""
        if not chat_history or len(chat_history) == 0:
            return StandaloneQuestion(question=latest_message)

        chat_history_str = messages_to_history_str(chat_history)
        logger.debug(chat_history_str)

        return self.client.chat.completions.create(
            model=self.model,
            response_model=StandaloneQuestion,
            messages=[
                {
                    "role": "system",
                    "content": CONDENSE_PROMPT_SYSTEM_TEMPLATE.format(
                        chat_history=chat_history_str,
                    ),
                },
                {
                    "role": "user",
                    "content": CONDENSE_PROMPT_USER_TEMPLATE.format(
                        question=latest_message
                    ),
                },
            ],
        )  # type: ignore

    def __call__(self, resolved_query: ResolvedQuery) -> EnhancedQuery:
        resolved_query.condensed_query = self.condense_question(
            resolved_query.chat_history, resolved_query.cleaned_query
        ).question
        enhanced_query = self.enhance_query(resolved_query)
        logger.debug(f"Openai query enhancements: {enhanced_query}")
        return enhanced_query


class QueryHandlerConfig(BaseSettings):
    default_query_clf_model: str = Field(
        ...,
        description="The name of the model to use for query classification",
        env="DEFAULT_QUERY_CLF_MODEL",
        validation_alias="default_query_clf_model",
    )
    fallback_query_clf_model: str = Field(
        "gpt-4-1106-preview",
        description="The name of the fallback model to use for query classification",
    )
    tokenizer: str = Field(
        "cl100k_base",
        description="The name of the tokenizer to use for query classification",
    )
    bot_name_pattern: str = Field(
        r"<@U[A-Z0-9]+>|@[a-zA-Z0-9\s]+\([a-zA-Z0-9\s]+\)|@[a-zA-Z0-9\s]+",
        description="The regex pattern to use for detecting bot names in queries",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )


class CohereQueryClassifier:
    def __init__(self, client: cohere.Client, model: str):
        self.client = client
        self.model = model

    def __call__(self, query: str) -> List[str]:
        response = self.client.classify(
            model=self.model,
            inputs=[query],
        )
        logger.debug(
            f"Cohere query classifications: {response.classifications}"
        )
        return response.classifications[0].predictions


class CompleteQuery(BaseModelV1):
    initial_query: str = Field(
        ...,
        description="The initial user query",
    )
    cleaned_query: str | None = Field(
        ...,
        description="The user query cleaned of bot names and other references",
    )
    condensed_query: str | None = Field(
        ...,
        description="The cleaned user query condensed to a standalone question",
    )
    intent_hints: str | None = Field(
        None,
        description="The descriptions of the intents related to the query",
    )
    language: str | None = Field(
        None,
        description="The language of the query",
    )
    chat_history: List[ChatMessage] | None = Field(
        None,
        description="The chat history of the query",
    )
    keywords: List[str] | None = Field(
        None,
        description="The keywords extracted and related to the query",
    )
    sub_queries: List[str] | None = Field(
        None,
        description="The sub-queries extracted from the query",
    )


class QueryHandler:
    def __init__(self, config: QueryHandlerConfig | None = None):
        self.config = (
            config
            if isinstance(config, QueryHandlerConfig)
            else QueryHandlerConfig()
        )
        self.tokenizer = tiktoken.get_encoding(self.config.tokenizer)
        self.bot_name_pattern = re.compile(self.config.bot_name_pattern)
        self.cohere_client = cohere.Client(os.environ["COHERE_API_KEY"])
        self.openai_client = OpenAI()
        self.query_classifier = CohereQueryClassifier(
            client=self.cohere_client, model=self.config.default_query_clf_model
        )
        self.query_enhancer = OpenaiQueryEnhancer(
            client=self.openai_client,
            model=self.config.fallback_query_clf_model,
        )

    def classify(self, query: str) -> List[str]:
        response = self.query_classifier(query)
        if not response or len(response) < 1:
            response = ["other"]
        return response

    def detect_language(self, query: str) -> str:
        response = self.cohere_client.detect_language(
            texts=[query],
        )
        return response.results[0].language_code

    def clean_query(self, query: str) -> str:
        cleaned_query = self.bot_name_pattern.sub("", query).strip()
        return cleaned_query

    def describe_query(self, query: str) -> str:
        classifications = self.classify(query)
        descriptions = []
        if not classifications:
            return "- " + INTENT_DESCRIPTIONS["other"]

        for classification in classifications:
            description = INTENT_DESCRIPTIONS.get(classification, "")
            descriptions.append(description)
        descriptions = "\n- ".join(descriptions)
        return descriptions

    def validate_and_format_question(self, question: str) -> str:
        """Validates and formats the given question.

        Args:
            question: A string representing the question to validate and format.

        Returns:
            A string representing the validated and formatted question.

        Raises:
            ValueError: If the question is too long.
        """
        question = " ".join(question.strip().split())
        question = self.clean_query(question)
        if len(self.tokenizer.encode(question)) > 1024:
            raise ValueError(
                f"Question is too long. Please rephrase your question to be shorter than {1024 * 3 // 4} words."
            )
        return question

    @staticmethod
    def combine_query_objects(
        resolved_query: ResolvedQuery, enhanced_query: EnhancedQuery
    ) -> CompleteQuery:
        descriptions = []
        if not enhanced_query.predicted_labels:
            intent_hints = "- " + INTENT_DESCRIPTIONS["other"]
        else:
            for classification in enhanced_query.predicted_labels:
                description = INTENT_DESCRIPTIONS.get(classification.label, "")
                descriptions.append(description)
            intent_hints = "\n- ".join(descriptions)

        keywords = [keyword.keyword for keyword in enhanced_query.keywords]
        sub_queries = [
            sub_query.query for sub_query in enhanced_query.sub_queries
        ]
        return CompleteQuery(
            initial_query=resolved_query.query,
            cleaned_query=resolved_query.cleaned_query,
            condensed_query=resolved_query.condensed_query,
            intent_hints=intent_hints,
            language=resolved_query.language,
            chat_history=resolved_query.chat_history,
            keywords=keywords,
            sub_queries=sub_queries,
        )

    def __call__(self, chat_request: ChatRequest) -> CompleteQuery:
        cleaned_query = self.validate_and_format_question(chat_request.question)
        chat_history = get_chat_history(chat_request.chat_history)
        language = self.detect_language(cleaned_query)
        if language == "en":
            intent = self.describe_query(
                cleaned_query,
            )
        else:
            intent = (
                "\n- "
                + INTENT_DESCRIPTIONS["other"]
                + " because the query is not in English"
            )
        resolved_query = ResolvedQuery(
            cleaned_query=cleaned_query,
            query=chat_request.question,
            intent=intent,
            language=language,
            chat_history=chat_history,
        )

        logger.debug(f"Resolved query: {resolved_query}")

        enhanced_query = self.query_enhancer(resolved_query)

        logger.debug(f"Enhanced query: {enhanced_query}")

        return self.combine_query_objects(resolved_query, enhanced_query)


def main():
    from wandbot.utils import Timer

    with Timer() as timer:
        config = QueryHandlerConfig()
        logger.info(config)
        query_handler = QueryHandler(config=QueryHandlerConfig())
        chat_request = ChatRequest(
            question="@wandbot (beta)When I execute run, where should I put requirement.txt in order to capture job?",
            chat_history=[],
            language="en",
            application="slack",
        )
        complete_query = query_handler(chat_request)
        print(complete_query.json(indent=2))
    print(f"Elapsed time: {timer.elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
