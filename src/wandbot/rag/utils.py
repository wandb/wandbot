import json
from pydantic import BaseModel

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate, format_document
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatLiteLLM

from litellm import completion_cost
from litellm.integrations.custom_logger import CustomLogger

from wandbot.utils import clean_document_content, get_logger

logger = get_logger(__name__)

class ChatModel:
    def __init__(self, temperature: float = 0.1, max_retries: int = 2):
        self.temperature = temperature
        self.max_retries = max_retries

    def __set_name__(self, owner, name):
        self.public_name = name
        self.private_name = "_" + name

    def __get__(self, obj, obj_type=None):
        value = getattr(obj, self.private_name)
        return value

    def __set__(self, obj, value):
        model = ChatLiteLLM(
            model_name=value,
            temperature=self.temperature,
            max_retries=self.max_retries,
            max_tokens=4096,
        )
        setattr(obj, self.private_name, model)


DEFAULT_QUESTION_PROMPT = PromptTemplate.from_template(
    template="""# Query

{page_content}

---

# Query Metadata

Language: {language}

Intents: 

{intents}
"""
)

# Sub-queries to consider answering: 

# {sub_queries}


def create_query_str(enhanced_query, document_prompt=DEFAULT_QUESTION_PROMPT):
    page_content = enhanced_query["standalone_query"]
    metadata = {
        "language": enhanced_query["language"],
        "intents": enhanced_query["intents"],
        # "sub_queries": "\t"
        # + "\n\t".join(enhanced_query["sub_queries"]).strip(),
    }
    doc = Document(page_content=page_content, metadata=metadata)
    doc = clean_document_content(doc)
    doc_string = format_document(doc, document_prompt)
    return doc_string


SIMPLE_DEFAULT_QUESTION_PROMPT = PromptTemplate.from_template(
    template="""# Query

{page_content}

---

# Query Metadata

Language: {language}
"""
)


def create_simple_query_str(data, document_prompt=SIMPLE_DEFAULT_QUESTION_PROMPT):
    page_content = data.question
    metadata = {
        "language": data.language,
    }
    doc = Document(page_content=page_content, metadata=metadata)
    doc = clean_document_content(doc)
    doc_string = format_document(doc, document_prompt)
    return doc_string


DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(
    template="source: {source}\nsource_type: {source_type}\nhas_code: {has_code}\n\n{page_content}"
)


def combine_documents(
    docs,
    document_prompt=DEFAULT_DOCUMENT_PROMPT,
    document_separator="\n\n---\n\n",
):
    cleaned_docs = [clean_document_content(doc) for doc in docs]
    doc_strings = [
        format_document(doc, document_prompt) for doc in cleaned_docs
    ]
    return document_separator.join(doc_strings)


SIMPLE_DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(
    template="source: {source}\nsource_type: {source_type}\nhas_code: {has_code}\n\n{page_content}"
)


def combine_simple_documents(
    docs,
    document_prompt=SIMPLE_DEFAULT_DOCUMENT_PROMPT,
    document_separator="\n\n---\n\n",
):
    cleaned_docs = [clean_document_content(doc) for doc in docs]
    doc_strings = [
        format_document(doc, document_prompt) for doc in cleaned_docs
    ]
    return document_separator.join(doc_strings)


def process_input_for_retrieval(retrieval_input):
    if isinstance(retrieval_input, list):
        retrieval_input = "\n".join(retrieval_input)
    elif isinstance(retrieval_input, dict):
        retrieval_input = json.dumps(retrieval_input)
    elif not isinstance(retrieval_input, str):
        retrieval_input = str(retrieval_input)
    return retrieval_input


def get_web_contexts(web_results):
    output_documents = []
    if not web_results:
        return []
    return (
        output_documents
        + [
            Document(
                page_content=document["context"], metadata=document["metadata"]
            )
            for document in web_results["web_context"]
        ]
        if web_results.get("web_context")
        else []
    )


class TotalsModel(BaseModel):
    completion_tokens: int = 0
    prompt_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0


class LiteLLMTokenCostLogger(CustomLogger):
    def __init__(self):
        # Use the Pydantic model for managing totals
        self.totals = TotalsModel()

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        # Update totals using the response object
        self.totals.completion_tokens += response_obj.usage.completion_tokens
        self.totals.prompt_tokens += response_obj.usage.prompt_tokens
        self.totals.total_tokens += response_obj.usage.total_tokens
        self.totals.total_cost += completion_cost(response_obj)
        logger.info(
            f"On Success: Completion Tokens: {response_obj.usage.completion_tokens}, "
            f"Prompt Tokens: {response_obj.usage.prompt_tokens}, "
            f"Total Tokens: {response_obj.usage.total_tokens}, "
            f"Cost: {completion_cost(response_obj)}"
        )

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        # Similar to the synchronous method, update totals for async events
        self.totals.completion_tokens += response_obj.usage.completion_tokens
        self.totals.prompt_tokens += response_obj.usage.prompt_tokens
        self.totals.total_tokens += response_obj.usage.total_tokens
        self.totals.total_cost += completion_cost(response_obj)
        logger.info(
            f"On Async Success: Completion Tokens: {response_obj.usage.completion_tokens}, "
            f"Prompt Tokens: {response_obj.usage.prompt_tokens}, "
            f"Total Tokens: {response_obj.usage.total_tokens}, "
            f"Cost: {completion_cost(response_obj)}"
        )

    def get_totals(self):
        current_totals = self.totals
        self.totals = TotalsModel()
        return current_totals
