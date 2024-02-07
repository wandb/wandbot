import os
from operator import itemgetter
from typing import List, Dict, Any

import requests
from langchain_core.runnables import (
    RunnableLambda,
    RunnableBranch,
    RunnableParallel,
    Runnable,
    RunnablePassthrough,
)
from pydantic import BaseModel, Field


class YouSearchResults(BaseModel):
    web_answer: str = Field("", description="response from you.com RAG model")
    web_context: List[Dict[str, Any]] = Field(
        [{}], description="context for the response"
    )


class YouSearch:
    def __init__(self, api_key: str, similarity_top_k: int = 10):
        self._api_key = api_key
        self.similarity_top_k = similarity_top_k

    def _rag(self, query: str) -> YouSearchResults:
        """Retrieve."""
        try:
            headers = {"X-API-Key": self._api_key}
            url = "https://api.ydc-index.io/rag"

            querystring = {
                "query": "Weights & Biases, W&B, wandb or Weave " + query,
                "num_web_results": self.similarity_top_k,
            }
            response = requests.get(url, headers=headers, params=querystring)
            if response.status_code != 200:
                return YouSearchResults()
            else:
                results = response.json()

            snippets = [hit["snippet"] for hit in results["hits"]]
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
                web_context=search_hits[: self.similarity_top_k],
            )
        except Exception as e:
            return YouSearchResults()

    def _retrieve(self, query: str) -> YouSearchResults:
        """Retrieve."""
        try:
            headers = {"X-API-Key": self._api_key}
            url = "https://api.ydc-index.io/search"

            querystring = {
                "query": "Weights & Biases, W&B, wandb or Weave " + query,
                "num_web_results": self.similarity_top_k,
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
                web_context=search_hits[: self.similarity_top_k],
            )
        except Exception as e:
            print(e)
            return YouSearchResults()

    def __call__(
        self, question: str, search_type: str = "rag"
    ) -> Dict[str, Any]:
        if search_type == "rag":
            web_results = self._rag(question)
        else:
            web_results = self._retrieve(question)
        return web_results.dict()


def load_web_answer_chain(search_field: str, top_k: int = 5) -> Runnable:

    you_search = YouSearch(os.environ["YOU_API_KEY"], top_k)

    web_answer_chain = RunnablePassthrough().assign(
        web_results=lambda x: you_search(
            question=x["question"], search_type=x["search_type"]
        )
    )

    branch = RunnableBranch(
        (
            lambda x: x["avoid"],
            RunnableLambda(lambda x: None),
        ),
        (
            lambda x: not x["avoid"],
            web_answer_chain | itemgetter("web_results"),
        ),
        RunnableLambda(lambda x: None),
    )

    return (
        RunnableParallel(
            question=itemgetter(search_field),
            search_type=itemgetter("search_type"),
            avoid=itemgetter("avoid_query"),
        )
        | branch
    )


def load_web_search_chain(search_field: str, top_k: int = 5) -> Runnable:

    you_search = YouSearch(os.environ["YOU_API_KEY"], top_k)

    web_answer_chain = RunnablePassthrough().assign(
        web_results=lambda x: you_search(
            question=x["question"], search_type=x["search_type"]
        )
    )

    branch = RunnableBranch(
        (
            lambda x: x["avoid"],
            RunnableLambda(lambda x: None),
        ),
        (
            lambda x: not x["avoid"],
            web_answer_chain | itemgetter("web_results"),
        ),
        RunnableLambda(lambda x: None),
    )

    return (
        RunnableParallel(
            question=itemgetter(search_field),
            search_type=itemgetter("search_type"),
            avoid=itemgetter("avoid_query"),
        )
        | branch
    )


def load_web_answer_enhancement_chain(top_k: int = 5) -> Runnable:
    web_answer_chain = load_web_answer_chain(
        search_field="standalone_question", top_k=top_k
    )

    return (
        RunnablePassthrough().assign(
            search_type=lambda x: "rag",
        )
        | web_answer_chain
    )


def load_web_search_enhancement_chain(top_k: int = 5) -> Runnable:
    web_answer_chain = load_web_answer_chain(
        search_field="standalone_question", top_k=top_k
    )

    return (
        RunnablePassthrough().assign(search_type=lambda x: "rag")
        | web_answer_chain
    )
