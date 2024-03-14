import json

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate, format_document
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatLiteLLM

from litellm import completion_cost
from litellm.integrations.custom_logger import CustomLogger

from wandbot.utils import clean_document_content

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

Sub-queries to consider answering: 

{sub_queries}
"""
)


def create_query_str(enhanced_query, document_prompt=DEFAULT_QUESTION_PROMPT):
    page_content = enhanced_query["standalone_query"]
    metadata = {
        "language": enhanced_query["language"],
        "intents": enhanced_query["intents"],
        "sub_queries": "\t"
        + "\n\t".join(enhanced_query["sub_queries"]).strip(),
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


class LiteLLMTokenCostLogger(CustomLogger):
    def __init__(self):
        self.reset_totals()  # Initialize totals

    def log_success_event(self, kwargs, response_obj, start_time, end_time): 
        self.completion_tokens += response_obj.usage.completion_tokens
        self.prompt_tokens += response_obj.usage.prompt_tokens
        self.total_tokens += response_obj.usage.total_tokens
        self.total_cost += completion_cost(response_obj)
        # Optionally, print each event's details
        print(f"On Success: {response_obj.usage}, Cost: {completion_cost(response_obj)}")

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        # Assuming response_obj will have similar structure in async
        self.completion_tokens += response_obj.usage.completion_tokens
        self.prompt_tokens += response_obj.usage.prompt_tokens
        self.total_tokens += response_obj.usage.total_tokens
        self.total_cost += completion_cost(response_obj)
        # Optionally, print each event's details
        print(f"On Success: {response_obj.usage}, Cost: {completion_cost(response_obj)}")

    def get_totals(self):
        """Returns the current totals and then resets the counters."""
        current_totals = {
            "completion_tokens": self.completion_tokens,
            "prompt_tokens": self.prompt_tokens,
            "total_tokens": self.total_tokens,
            "cost": self.total_cost
        }
        self.reset_totals()  # Reset totals after fetching
        return current_totals

    def reset_totals(self):
        """Resets the total counts and costs to zero."""
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.total_tokens = 0
        self.total_cost = 0.0
