from langchain.retrievers.document_compressors import CohereRerank
from langchain_core.runnables import RunnableBranch


class CohereRerankChain:
    def __set_name__(self, owner, name):
        self.public_name = name
        self.private_name = "_" + name

    def __get__(self, obj, obj_type=None):
        if getattr(obj, "top_k") is None:
            raise AttributeError("Top k must be set before using rerank chain")

        def load_rerank_chain(language):
            if language == "en":
                cohere_rerank = CohereRerank(
                    top_n=obj.top_k, model="rerank-english-v2.0"
                )
            else:
                cohere_rerank = CohereRerank(
                    top_n=obj.top_k, model="rerank-multilingual-v2.0"
                )

            return lambda x: cohere_rerank.compress_documents(
                documents=x["context"], query=x["question"]
            )

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
