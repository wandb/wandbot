import weave
# from langchain_cohere import CohereRerank
from langchain_core.runnables import RunnableBranch

import weave
from langchain_core.runnables import RunnableBranch
from langchain_core.documents import Document
from copy import deepcopy
import cohere
import os


class CohereRerankChain:
    def __set_name__(self, owner, name):
        self.public_name = name
        self.private_name = "_" + name

    def __set__(self, obj, value):
        self.english_model = value["english_reranker_model"]
        self.multilingual_model = value["multilingual_reranker_model"]

    def __get__(self, obj, obj_type=None):
        if getattr(obj, "top_k") is None:
            raise AttributeError("Top k must be set before using rerank chain")

        co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))

        @weave.op
        def load_rerank_chain(language):
            model = self.english_model if language == "en" else self.multilingual_model
            
            return lambda x: [
                Document(
                    page_content=x["context"][hit.index].page_content,
                    metadata={
                        **deepcopy(x["context"][hit.index].metadata),
                        "relevance_score": hit.relevance_score
                    }
                )
                for hit in co.rerank(
                    query=x["question"],
                    documents=[doc.page_content for doc in x["context"]],
                    top_n=obj.top_k,
                    model=model
                ).results
            ]

        cohere_rerank = RunnableBranch(
            (
                lambda x: x["language"] == "en",
                load_rerank_chain("en"),
            ),
            (
                lambda x: x["language"],
                load_rerank_chain("ja"),
            ),
            load_rerank_chain("ja"),
        )

        return cohere_rerank


# class CohereRerankChain:
#     def __set_name__(self, owner, name):
#         self.public_name = name
#         self.private_name = "_" + name

#     def __set__(self, obj, value):
#         self.english_model = value["english_reranker_model"]
#         self.multilingual_model = value["multilingual_reranker_model"]

#     def __get__(self, obj, obj_type=None):
#         if getattr(obj, "top_k") is None:
#             raise AttributeError("Top k must be set before using rerank chain")

#         @weave.op
#         def load_rerank_chain(language):
#             if language == "en":
#                 cohere_rerank = CohereRerank(
#                     top_n=obj.top_k, model=self.english_model
#                 )
#             else:
#                 cohere_rerank = CohereRerank(
#                     top_n=obj.top_k, model=self.multilingual_model
#                 )

#             return lambda x: cohere_rerank.compress_documents(
#                 documents=x["context"], query=x["question"]
#             )

#         cohere_rerank = RunnableBranch(
#             (
#                 lambda x: x["language"] == "en",
#                 load_rerank_chain("en"),
#             ),
#             (
#                 lambda x: x["language"],
#                 load_rerank_chain("ja"),
#             ),
#             load_rerank_chain("ja"),
#         )

#         return cohere_rerank
