import asyncio
import os
import traceback
import sys
from typing import Any, Dict, List

import weave

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from wandbot.utils import get_logger, ErrorInfo, get_error_file_path
from wandbot.schema.document import Document
from wandbot.schema.api_status import APIStatus

logger = get_logger(__name__)

dotenv_path = os.path.join(os.path.dirname(__file__), "../../../.env")
load_dotenv(dotenv_path)


class WebSearchResults(BaseModel):
    api_status: APIStatus
    web_contexts: List[Document]


class YouSearchResults(BaseModel):
    web_answer: str = Field("", description="response from you.com RAG model")
    web_context: List[Dict[str, Any]] = Field(
        [{}], description="context for the response"
    )
    api_status: APIStatus = Field(..., description="Status of the API call")


class YouSearchConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="allow"
    )
    you_api_key: str = Field(
        ...,
        description="API key for you.com search API",
        env="YOU_API_KEY",
        validation_alias="you_api_key",
    )
    top_k: int = Field(
        10,
        description="Number of top k results to retrieve from you.com",
    )
    search_type: str = Field(
        "rag",
        description="Type of search to perform. Options: rag, retrieve",
    )


class YouSearch:
    config: YouSearchConfig = YouSearchConfig()

    def __init__(self, config: YouSearchConfig = None):
        if config is not None:
            self.config = config

    def _rag(self, query: str) -> YouSearchResults:
        """Retrieve."""
        error_info = ErrorInfo(component="web_search")
        try:
            headers = {"X-API-Key": self.config.you_api_key}
            url = "https://api.ydc-index.io/rag"

            querystring = {
                "query": "Answer the following question in the context of Weights & Biases, W&B, wandb and/or Weave\n"
                + query,
                "num_web_results": self.config.top_k,
                "safesearch": "strict",
            }
            response = requests.get(url, headers=headers, params=querystring)
            if response.status_code != 200:
                error_msg = f"Error in RAG web search, status code: {response.status_code}"
                logger.error(error_msg)
                error_info.has_error = True
                error_info.error_message = error_msg
                error_info.error_type = "HTTPError"
                return YouSearchResults(
                    api_status=APIStatus(
                        component="web_search",
                        success=False,
                        error_info=error_info
                    )
                )
            elif response.json()["error_code"].lower() == "payment required":
                error_msg = f"Error in RAG web search, error code: {response.json()['error_code']}"
                logger.error(error_msg)
                error_info.has_error = True
                error_info.error_message = error_msg
                error_info.error_type = "PaymentRequired"
                return YouSearchResults(
                    api_status=APIStatus(
                        component="web_search",
                        success=False,
                        error_info=error_info
                    )
                )
            else:
                results = response.json()
                logger.info(f"RAG web search results: {results}")

            snippets = [
                f'Title: {hit["title"]}\nDescription: {hit["description"]}\n{hit["snippet"]}'
                for hit in results["hits"]
            ]
            snippet_metadata = [
                {
                    "source": hit["url"],
                    "language": "en",
                    "description": hit["description"],
                    "title": hit["title"],
                    "tags": ["you.com"],
                    "source_type": "web_search",
                    "has_code": None,
                }
                for hit in results["hits"]
            ]
            search_hits = []
            for snippet, metadata in zip(snippets, snippet_metadata):
                search_hits.append({"context": snippet, "metadata": metadata})

            return YouSearchResults(
                web_answer=results["answer"],
                web_context=search_hits[: self.config.top_k],
                api_status=APIStatus(
                    component="web_search",
                    success=True
                )
            )
        except Exception as e:
            error_msg = f"Error in RAG web search: {e}"
            logger.error(error_msg)
            error_info.has_error = True
            error_info.error_message = error_msg
            error_info.error_type = type(e).__name__
            error_info.stacktrace = ''.join(traceback.format_exc())
            error_info.file_path = get_error_file_path(sys.exc_info()[2])
            return YouSearchResults(
                api_status=APIStatus(
                    component="web_search",
                    success=False,
                    error_info=error_info
                )
            )

    def _retrieve(self, query: str) -> YouSearchResults:
        """Retrieve."""
        try:
            headers = {"X-API-Key": self.config.you_api_key}
            url = "https://api.ydc-index.io/search"

            querystring = {
                "query": "Weights & Biases, W&B, wandb or Weave " + query,
                "num_web_results": self.config.top_k,
            }
            response = requests.get(url, headers=headers, params=querystring)
            if response.status_code != 200:
                logger.error(
                    f"Error in retrieve web search, Status code: {response.status_code}"
                )
                return YouSearchResults(success=False)
            elif response.json()["error_code"].lower() == "payment required":
                logger.error(
                    f"Error in retrieve web search, error code: {response.json()['error_code']}"
                )
                return YouSearchResults(success=False)
            else:
                results = response.json()

            snippets = [hit["snippets"] for hit in results["hits"]]
            snippet_metadata = [
                {
                    "source": hit["url"],
                    "language": "en",
                    "description": hit["description"],
                    "title": hit["title"],
                    "tags": ["you.com"],
                    "source_type": "web_search",
                    "has_code": None,
                }
                for hit in results["hits"]
            ]
            search_hits = []
            for snippet_list, metadata in zip(snippets, snippet_metadata):
                for snippet in snippet_list:
                    search_hits.append(
                        {"context": snippet, "metadata": metadata}
                    )

            return YouSearchResults(
                web_answer="",
                web_context=search_hits[: self.config.top_k],
                success=True,
            )
        except Exception as e:
            logger.error(f"Error in retrieve web search: {e}")
            return YouSearchResults(success=False)

    def __call__(
        self,
        question: str,
    ) -> YouSearchResults:
        if self.config.search_type == "rag":
            return self._rag(question)
        else:
            return self._retrieve(question)

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

@weave.op
def run_web_search(query: str, top_k: int, avoid=False) -> WebSearchResults:
    try:
        if avoid:
            logger.debug(f"Skipping web search, avoid: {avoid}")
            return WebSearchResults(
                api_status=APIStatus(
                    component="web_search",
                    success=False,
                    error_info=ErrorInfo(
                        component="web_search",
                        has_error=False,
                        error_message="Web search is disabled",
                        error_type="WebSearchDisabled"
                    )
                ),
                web_contexts=[],
            )
        
        # Run web search
        yousearch = YouSearch(YouSearchConfig())
        web_results = yousearch(query)
        if web_results.api_status.success:
            web_contexts = get_web_contexts(web_results)[:top_k]
        else:
            logger.debug(
                f"Issue running web search, web_results: {web_results}"
            )
            web_contexts = []
        return WebSearchResults(
            api_status=web_results.api_status,
            web_contexts=web_contexts,
        )
    except Exception as e:
        error_info = ErrorInfo(
            component="web_search",
            has_error=True,
            error_message=str(e),
            error_type=type(e).__name__,
            stacktrace=''.join(traceback.format_exc()),
            file_path=get_error_file_path(sys.exc_info()[2])
        )
        logger.error(f"Error running web search: {e}")
        return WebSearchResults(
            api_status=APIStatus(
                component="web_search",
                success=False,
                error_info=error_info
            ),
            web_contexts=[],
        )
    
async def _async_run_web_search(query: str, top_k: int, avoid: bool) -> WebSearchResults:
    return await asyncio.to_thread(
        run_web_search,
        query=query,
        top_k=top_k,
        avoid=avoid
    )