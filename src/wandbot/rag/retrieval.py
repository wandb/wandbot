from typing import List

from langchain.retrievers.document_compressors import CohereRerank
from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnablePassthrough
from wandbot.rag.utils import get_web_contexts
from wandbot.retriever.base import VectorStore
from wandbot.retriever.web_search import YouSearch, YouSearchConfig


def reciprocal_rank_fusion(results: list[list[Document]], k=60):
    text_to_doc = {}
    fused_scores = {}
    for docs in results:
        # Assumes the docs are returned in sorted order of relevance
        for rank, doc in enumerate(docs):
            doc_content = doc.page_content
            text_to_doc[doc_content] = doc
            if doc_content not in fused_scores:
                fused_scores[doc_content] = 0.0
            fused_scores[doc_content] += 1 / (rank + k)

    ranked_results = dict(
        sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    )

    ranked_results = [text_to_doc[text] for text in ranked_results.keys()]
    return ranked_results


def run_web_search(query, avoid=False) -> list:
    if avoid:
        return []
    yousearch = YouSearch(YouSearchConfig())
    web_results = yousearch(query)
    return get_web_contexts(web_results)


def rerank_results(
    queries: List[str],
    context: List[Document],
    top_k: int = 5,
    language: str = "en",
):
    if language == "en":
        reranker = CohereRerank(top_n=top_k, model="rerank-english-v2.0")
    else:
        reranker = CohereRerank(top_n=top_k, model="rerank-multilingual-v2.0")

    query = "\n".join(queries)
    ranked_results = reranker.compress_documents(documents=context, query=query)
    return ranked_results


class FusionRetrieval:
    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = 5,
        search_type: str = "mmr",
    ):
        self.vectorstore = vector_store
        self.top_k = top_k
        self.search_type = search_type

        self.retriever = self.vectorstore.as_parent_retriever(
            search_type=self.search_type, search_kwargs={"k": self.top_k}
        )

        self._chain = None

    @property
    def chain(self) -> Runnable:
        if self._chain is None:
            self._chain = (
                RunnablePassthrough().assign(
                    docs_context=lambda x: self.retriever.batch(
                        x["all_queries"]
                    ),
                    web_context=lambda x: run_web_search(
                        x["standalone_query"], x["avoid_query"]
                    ),
                )
                | RunnablePassthrough().assign(
                    full_context=lambda x: x["docs_context"]
                    + [x["web_context"]]
                )
                | RunnablePassthrough().assign(
                    fused_context=lambda x: reciprocal_rank_fusion(
                        x["full_context"]
                    )
                )
                | RunnablePassthrough().assign(
                    context=lambda x: rerank_results(
                        [x["standalone_query"]],
                        x["fused_context"],
                        self.top_k,
                        x["language"],
                    )
                )
            )
        return self._chain
