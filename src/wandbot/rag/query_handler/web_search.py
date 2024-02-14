from operator import itemgetter
from typing import Any, Dict, List

import requests
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class YouSearchResults(BaseModel):
    web_answer: str = Field("", description="response from you.com RAG model")
    web_context: List[Dict[str, Any]] = Field(
        [{}], description="context for the response"
    )


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
                return YouSearchResults()
            else:
                results = response.json()

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
            )
        except Exception as e:
            return YouSearchResults()

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
                return YouSearchResults()
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
            )
        except Exception as e:
            print(e)
            return YouSearchResults()

    def __call__(
        self,
        question: str,
    ) -> Dict[str, Any]:
        if self.config.search_type == "rag":
            web_results = self._rag(question)
        else:
            web_results = self._retrieve(question)
        return web_results.dict()


class YouWebRagSearchEnhancer:
    def __init__(self):
        self.you_search = YouSearch()
        self._chain = None

    @property
    def chain(self) -> Runnable:
        if self._chain is None:
            search_chain = RunnablePassthrough().assign(
                web_results=lambda x: self.you_search(question=x["question"])
            )

            branch = RunnableBranch(
                (
                    lambda x: x["avoid"],
                    RunnableLambda(lambda x: None),
                ),
                (
                    lambda x: not x["avoid"],
                    search_chain | itemgetter("web_results"),
                ),
                RunnableLambda(lambda x: None),
            )

            self._chain = (
                RunnableParallel(
                    question=itemgetter("standalone_question"),
                    avoid=itemgetter("avoid_query"),
                )
                | branch
            )
        return self._chain
