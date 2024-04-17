import enum
from operator import itemgetter
from typing import List

import cohere
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI
from pydantic.v1 import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from wandbot.rag.utils import ChatModel


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


class Label(BaseModel):
    "An intent label to be associated with the query"

    reasoning: str = Field(
        ...,
        description="The reason for the identifying the intent",
    )
    label: Labels = Field(
        ..., description="An intent associated with the query"
    )


class MultiLabel(BaseModel):
    "A list of intents associated with the query"
    intents: List[Label] = Field(
        ...,
        description="The list of intents associated with the query",
        min_items=1,
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


def get_intent_descriptions(intents: List[str]) -> str:
    descriptions = []
    if not intents:
        return "- " + INTENT_DESCRIPTIONS["other"]

    for classification in intents:
        description = INTENT_DESCRIPTIONS.get(classification, "")
        descriptions.append(description)
    descriptions = "- " + "\n- ".join(descriptions)
    return descriptions


def get_intent_hints(intents: List[str]) -> str:
    descriptions = []
    if not intents:
        return "- " + QUERY_INTENTS["other"]

    for classification in intents:
        description = QUERY_INTENTS.get(classification, "")
        descriptions.append(description)
    descriptions = "- " + "\n- ".join(descriptions)
    return descriptions


class CohereClassifierConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="allow"
    )
    cohere_api_key: str = Field(
        ...,
        description="The API key for the Cohere API",
        env="COHERE_API_KEY",
        validation_alias="cohere_api_key",
    )
    cohere_query_clf_model: str = Field(
        ...,
        description="The fine-tuned cohere model to use for classification",
        env="COHERE_QUERY_CLF_MODEL",
        validation_alias="cohere_query_clf_model",
    )


class CohereQueryClassifier:
    config: CohereClassifierConfig = CohereClassifierConfig()

    def __init__(self) -> None:
        self.client = cohere.Client(self.config.cohere_api_key)

    def __call__(self, query: str) -> str:
        response = self.client.classify(
            model=self.config.cohere_query_clf_model,
            inputs=[query],
        )
        return get_intent_descriptions(response.classifications[0].predictions)


intents_descriptions_str = "\n".join(
    [
        f"{label}:\t{description}"
        for label, description in INTENT_DESCRIPTIONS.items()
    ]
)

INTENT_PROMPT_MESSAGES = [
    (
        "system",
        """You are a Weights & Biases support manager. Your goal is to enhance the query by identifying one or more intents related to the query.
        Here is the mapping of the intents and their descriptions.
        """
        + intents_descriptions_str,
    ),
    ("human", "Enhance the following user query:\n{question}"),
    (
        "human",
        "Here is an initial list of intent hints that maybe relevant:\n{intent_hints}",
    ),
    ("human", "Tip: Make sure to answer in the correct format"),
]


def check_avoid_intent(intents: List[str]) -> bool:
    return any(
        [
            intent
            in [
                Labels.NEFARIOUS_QUERY.value,
                Labels.OPINION_REQUEST.value,
                Labels.NEEDS_MORE_INFO.value,
                Labels.UNRELATED.value,
            ]
            for intent in intents
        ]
    )


class IntentsEnhancer:
    model: ChatModel = ChatModel()
    fallback_model: ChatModel = ChatModel(max_retries=6)

    def __init__(
        self,
        model: str = "gpt-4-1106-preview",
        fallback_model: str = "gpt-4-1106-preview",
    ):
        self.model = model
        self.fallback_model = fallback_model

        self.cohere_classifier = CohereQueryClassifier()
        self.prompt = ChatPromptTemplate.from_messages(INTENT_PROMPT_MESSAGES)
        self._chain = None

    @property
    def chain(self) -> Runnable:
        if self._chain is None:
            base_chain = self._load_chain(self.model)
            fallback_chain = self._load_chain(self.fallback_model)
            self._chain = base_chain.with_fallbacks([fallback_chain])

        return self._chain

    def _load_chain(self, model: ChatOpenAI) -> Runnable:
        # load the cohere classifier chain

        cohere_classify_chain = RunnablePassthrough.assign(
            intent_hints=lambda x: self.cohere_classifier(x["question"])
        )

        # load the intent extraction chain
        intents_classification_chain = create_structured_output_runnable(
            MultiLabel, model, self.prompt
        )

        intent_extraction_chain = (
            cohere_classify_chain
            | intents_classification_chain
            | RunnableLambda(
                lambda x: [intent.label.value for intent in x.intents]
            )
            | RunnableParallel(
                intent_hints=get_intent_hints,
                intent_labels=RunnablePassthrough(),
            )
        )

        # load the intent enhancement chain
        intent_enhancement_chain = RunnableParallel(
            question=itemgetter("question"),
            standalone_question=itemgetter("standalone_question"),
            chat_history=itemgetter("chat_history"),
            language=itemgetter("language"),
            intents=(
                {"question": itemgetter("standalone_question")}
                | intent_extraction_chain
            ),
        ) | RunnablePassthrough.assign(
            avoid_query=lambda x: check_avoid_intent(
                x["intents"]["intent_labels"]
            )
        )

        return intent_enhancement_chain
