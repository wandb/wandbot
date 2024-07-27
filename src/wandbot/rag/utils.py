import json

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate, format_document
from langchain_openai import ChatOpenAI

from wandbot.utils import clean_document_content
from wandbot.retriever.web_search import YouSearchResults


class ChatModel:
    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries

    def __set_name__(self, owner, name):
        self.public_name = name
        self.private_name = "_" + name

    def __get__(self, obj, obj_type=None):
        value = getattr(obj, self.private_name)
        return value

    def __set__(self, obj, value):
        model = ChatOpenAI(
            model_name=value["model_name"],
            temperature=value["temperature"],
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


def process_input_for_retrieval(retrieval_input):
    if isinstance(retrieval_input, list):
        retrieval_input = "\n".join(retrieval_input)
    elif isinstance(retrieval_input, dict):
        retrieval_input = json.dumps(retrieval_input)
    elif not isinstance(retrieval_input, str):
        retrieval_input = str(retrieval_input)
    return retrieval_input


def get_web_contexts(web_results: YouSearchResults):
    output_documents = []
    if not web_results:
        return []
    return (
        output_documents
        + [
            Document(
                page_content=document["context"], metadata=document["metadata"]
            )
            for document in web_results.web_context
        ]
        if web_results.web_context
        else []
    )
