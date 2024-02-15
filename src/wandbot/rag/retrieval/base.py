from operator import itemgetter

from langchain.load import dumps, loads
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from wandbot.ingestion.config import VectorStoreConfig
from wandbot.rag.utils import get_web_contexts, process_input_for_retrieval
from wandbot.retriever import OpenAIEmbeddingsModel, VectorStore
from wandbot.retriever.reranking import CohereRerankChain


def reciprocal_rank_fusion(results: list[list], k=60):
    fused_scores = {}
    for docs in results:
        # Assumes the docs are returned in sorted order of relevance
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            previous_score = fused_scores[doc_str]
            fused_scores[doc_str] += 1 / (rank + k)

    ranked_results = [
        (loads(doc), score)
        for doc, score in sorted(
            fused_scores.items(), key=lambda x: x[1], reverse=True
        )
    ]
    return [item[0] for item in ranked_results]


class RagRetrievalChain:
    def __init__(self, field: str = "question"):
        self.field = field

    def __set_name__(self, owner, name):
        self.public_name = name
        self.private_name = "_" + name

    def __get__(self, obj, obj_type=None):
        if getattr(obj, "retriever") is None:
            raise AttributeError(
                "Retriever must be set before setting retrieval chain"
            )
        default_input_chain = (
            itemgetter("standalone_question")
            | RunnablePassthrough()
            | process_input_for_retrieval
            | RunnableParallel(context=obj.retriever)
            | itemgetter("context")
        )

        input_chain = (
            itemgetter(self.field)
            | RunnablePassthrough()
            | process_input_for_retrieval
            | RunnableParallel(context=obj.retriever)
            | itemgetter("context")
        )

        retrieval_chain = RunnableBranch(
            (
                lambda x: not x["avoid_query"],
                input_chain,
            ),
            (
                lambda x: x["avoid_query"],
                default_input_chain,
            ),
            default_input_chain,
        )
        return retrieval_chain


class FusionRetrieval:
    question_chain = RagRetrievalChain("question")
    standalone_question_chain = RagRetrievalChain("standalone_question")
    keywords_chain = RagRetrievalChain("keywords")
    vector_search_chain = RagRetrievalChain("vector_search")
    web_context_chain = RunnableLambda(
        lambda x: get_web_contexts(x["web_results"])
    )
    cohere_rerank_chain = CohereRerankChain()
    embeddings_model: OpenAIEmbeddingsModel = OpenAIEmbeddingsModel(
        dimensions=768
    )

    def __init__(
        self,
        vector_store_config: VectorStoreConfig,
        top_k=5,
        search_type="mmr",
    ):
        self.vector_store = VectorStore.from_config(vector_store_config)

        self.retriever = self.vector_store.as_parent_retriever(
            search_type=search_type, search_kwargs={"k": top_k * 4}
        )
        self.embeddings_model = vector_store_config.embeddings_model  # type: ignore
        self.top_k = top_k
        self.redundant_filter = EmbeddingsRedundantFilter(
            embeddings=self.embeddings_model
        ).transform_documents

        self._chain = None

    @property
    def chain(self) -> Runnable:
        if self._chain is None:
            combined_retrieval_chain = (
                RunnableParallel(
                    question=self.question_chain,
                    standalone_question=self.standalone_question_chain,
                    keywords=self.keywords_chain,
                    vector_search=self.vector_search_chain,
                    web_context=self.web_context_chain,
                )
                | itemgetter(
                    "question",
                    "standalone_question",
                    "keywords",
                    "vector_search",
                    "web_context",
                )
                | reciprocal_rank_fusion
                | self.redundant_filter
            )

            self._chain = (
                RunnableParallel(
                    context=combined_retrieval_chain,
                    question=itemgetter("question"),
                    language=itemgetter("language"),
                )
                | self.cohere_rerank_chain
            )

        return self._chain
