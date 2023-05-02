import re
from typing import Any, Dict, List, Optional

from langchain import BasePromptTemplate, LLMChain
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.conversational_retrieval.base import (
    BaseConversationalRetrievalChain,
)
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.question_answering import load_qa_chain
from langchain.retrievers import TFIDFRetriever
from langchain.schema import BaseLanguageModel, BaseRetriever, Document
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
            search_kwargs={"k": 3},
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


def deduplicate_docs_with_order(docs: List[Document]) -> List[Document]:
    deduplicated = []
    seen_ids = {}
    for doc in docs:
        doc_id = doc.metadata.get("doc_id")
        if doc_id not in seen_ids:
            deduplicated.append(doc)
            seen_ids[doc_id] = True
    return deduplicated


class HybridRetriever(BaseRetriever, BaseModel):
    dense: VectorStoreRetrieverWithScore
    sparse: TFIDFRetriever
    k: int = 6

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def get_relevant_documents(self, query: str) -> List[Document]:
        self.dense.search_kwargs = {"k": self.k // 2}
        self.sparse.k = self.k // 2

        chroma_results = self.dense.get_relevant_documents(query)
        tfidf_results = self.sparse.get_relevant_documents(query)
        all_results = chroma_results + tfidf_results
        results = deduplicate_docs_with_order(all_results)
        return results

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError("This method is not implemented for this retriever.")


class ConversationalRetrievalQAWithSourcesandScoresChain(ConversationalRetrievalChain):
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

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        retriever: BaseRetriever,
        condense_question_prompt: BasePromptTemplate = CONDENSE_QUESTION_PROMPT,
        qa_prompt: Optional[BasePromptTemplate] = None,
        chain_type: str = "stuff",
        **kwargs: Any,
    ) -> BaseConversationalRetrievalChain:
        """Load chain from LLM."""
        if chain_type == "map_reduce":
            doc_chain = load_qa_chain(
                llm,
                chain_type=chain_type,
                question_prompt=qa_prompt,
            )
        else:
            doc_chain = load_qa_chain(
                llm,
                chain_type=chain_type,
                prompt=qa_prompt,
            )
        condense_question_chain = LLMChain(llm=llm, prompt=condense_question_prompt)
        return cls(
            retriever=retriever,
            combine_docs_chain=doc_chain,
            question_generator=condense_question_chain,
            **kwargs,
        )

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys.

        :meta private:
        """
        _output_keys = [self.output_key]
        if self.return_source_documents:
            _output_keys = _output_keys + ["source_documents"] + ["sources"]
        return _output_keys

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        results = super()._call(inputs)
        answer = results["answer"]
        if re.search(r"Sources:\s", answer, flags=re.IGNORECASE):
            answers_and_sources = re.split(r"Sources:\s", answer, flags=re.IGNORECASE)
            if len(answers_and_sources) > 1:
                answer = answers_and_sources[0]
                sources = answers_and_sources[1]
            else:
                sources = ""
        else:
            sources = ""
        results["answer"] = answer
        results["sources"] = sources
        return results
