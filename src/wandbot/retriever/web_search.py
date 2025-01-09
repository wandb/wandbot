import os
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from wandbot.utils import get_logger

logger = get_logger(__name__)

dotenv_path = os.path.join(os.path.dirname(__file__), "../../../.env")
load_dotenv(dotenv_path)


class WebSearchResults(BaseModel):
    web_search_success: bool
    web_contexts: List


class YouSearchResults(BaseModel):
    web_answer: str = Field("", description="response from you.com RAG model")
    web_context: List[Dict[str, Any]] = Field(
        [{}], description="context for the response"
    )
    success: bool = Field(..., description="success of the web search")


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
                logger.error(
                    f"Error in RAG web search, status code: {response.status_code}"
                )
                return YouSearchResults(success=False)
            elif response.json()["error_code"].lower() == "payment required":
                logger.error(
                    f"Error in RAG web search, error code: {response.json()['error_code']}"
                )
                return YouSearchResults(success=False)
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
                success=True,
            )
        except Exception as e:
            logger.error(f"Error in RAG web search: {e}")
            return YouSearchResults(success=False)

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
