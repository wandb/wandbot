from operator import itemgetter

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnableParallel

from wandbot.chat.response_synthesis import load_response_synthesizer_chain
from wandbot.ingestion.config import VectorStoreConfig
from wandbot.query_handler.query_enhancer import load_query_enhancement_chain
from wandbot.retriever.base import (
    load_retriever_with_options,
    load_vector_store_from_config,
)
from wandbot.retriever.fusion import load_fusion_retriever_chain


def load_rag_chain(
    model,
    fallback_model,
    embeddings_model,
    lang_detect_path,
    vector_store_path,
    search_type,
    top_k=10,
):
    fallback_query_enhancer_chain = load_query_enhancement_chain(
        fallback_model, lang_detect_path
    )
    query_enhancer_chain = load_query_enhancement_chain(
        model, lang_detect_path
    ).with_fallbacks([fallback_query_enhancer_chain])

    vectorstore_config = VectorStoreConfig(persist_dir=vector_store_path)
    vectorstore = load_vector_store_from_config(vectorstore_config)
    base_retriever = load_retriever_with_options(
        vectorstore, search_type=search_type, search_kwargs={"top_k": top_k * 2}
    )
    parent_retriever = base_retriever | RunnableLambda(
        lambda docs: [
            Document(
                page_content=doc.metadata.get(
                    "source_content", doc.page_content
                ),
                metadata=doc.metadata,
            )
            for doc in docs
        ]
    )

    fallback_response_synthesis_chain = load_response_synthesizer_chain(
        fallback_model
    )
    response_synthesis_chain = load_response_synthesizer_chain(
        model
    ).with_fallbacks([fallback_response_synthesis_chain])

    ranked_retrieval_chain = load_fusion_retriever_chain(
        parent_retriever, embeddings=embeddings_model, top_k=top_k
    )

    rag_chain = (
        RunnableParallel(
            query=query_enhancer_chain.with_config(
                {"run_name": "query_enhancer"}
            )
        )
        | RunnableParallel(
            query=itemgetter("query"),
            context=lambda x: itemgetter("query") | ranked_retrieval_chain,
        ).with_config({"run_name": "retrieval"})
        | RunnableParallel(
            query=itemgetter("query"),
            context=RunnableLambda(
                lambda x: [
                    {"page_content": p.page_content, "metadata": p.metadata}
                    for p in x["context"]
                ]
            ),
            result=response_synthesis_chain,
        ).with_config({"run_name": "response_synthesis"})
    )

    return rag_chain
