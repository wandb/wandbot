from typing import Any, Dict, List

from langchain.chains import ConversationalRetrievalChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.retrievers import TFIDFRetriever
from langchain.schema import BaseRetriever, Document
from langchain.vectorstores import Chroma
from langchain.vectorstores.base import VectorStoreRetriever
from pydantic import BaseModel


class VectorStoreRetrieverWithScore(VectorStoreRetriever):
    vectorstore: Chroma

    def get_relevant_documents(self, query: str) -> List[Document]:
        if self.search_type == "similarity":
            docs_and_scores = self.vectorstore.similarity_search_with_score(
                query, **self.search_kwargs
            )
            docs = []
            for doc, score in docs_and_scores:
                doc.metadata["score"] = score
                docs.append(doc)
        elif self.search_type == "mmr":
            docs = self.vectorstore.max_marginal_relevance_search(
                query, **self.search_kwargs
            )
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs


class ChromaWithEmbeddingsAndScores(Chroma):
    def add_texts_and_embeddings(self, documents, embeddings, ids, metadatas):
        self._collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )

    def as_retriever(self) -> VectorStoreRetrieverWithScore:
        return VectorStoreRetrieverWithScore(
            vectorstore=self,
            search_type="similarity",
            search_kwargs={"k": 4},
        )


class TFIDFRetrieverWithScore(TFIDFRetriever):
    def get_relevant_documents(self, query: str) -> List[Document]:
        from sklearn.metrics.pairwise import cosine_similarity

        query_vec = self.vectorizer.transform(
            [query]
        )  # Ip -- (n_docs,x), Op -- (n_docs,n_Feats)
        results = cosine_similarity(self.tfidf_array, query_vec).reshape(
            (-1,)
        )  # Op -- (n_docs,1) -- Cosine Sim with each doc
        return_docs = []
        for i in results.argsort()[-self.k :][::-1]:
            doc = self.docs[i]
            doc.metadata["score"] = results[i]
            return_docs.append(doc)
        return return_docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError("This method is not implemented for this retriever.")


class HybridRetriever(BaseRetriever, BaseModel):
    dense: VectorStoreRetrieverWithScore
    sparse: TFIDFRetriever

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def get_relevant_documents(self, query: str) -> List[Document]:
        chroma_results = self.dense.get_relevant_documents(query)
        tfidf_results = self.sparse.get_relevant_documents(query)
        return chroma_results + tfidf_results

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError("This method is not implemented for this retriever.")


class ConversationalRetrievalQAWithSourcesChainWithScore(ConversationalRetrievalChain):
    reduce_k_below_max_tokens: bool = True
    max_tokens_limit: int = 2816

    def _reduce_tokens_below_limit(self, docs: List[Document]) -> List[Document]:
        num_docs = len(docs)

        if self.max_tokens_limit and isinstance(
            self.combine_docs_chain, StuffDocumentsChain
        ):
            tokens = [
                self.combine_docs_chain.llm_chain.llm.get_num_tokens(doc.page_content)
                for doc in docs
            ]
            token_count = sum(tokens[:num_docs])
            while token_count > self.max_tokens_limit:
                num_docs -= 1
                token_count -= tokens[num_docs]

        return docs[:num_docs]

    def _get_docs(self, question: str, inputs: Dict[str, Any]) -> List[Document]:
        docs = self.retriever.get_relevant_documents(question)
        return self._reduce_tokens_below_limit(docs)
