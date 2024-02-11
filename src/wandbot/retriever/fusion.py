import json
from operator import itemgetter

from langchain.load import dumps, loads
from langchain.prompts.prompt import PromptTemplate
from langchain.retrievers.document_compressors import CohereRerank
from langchain.schema import Document, format_document
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

from wandbot.utils import clean_document_content

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(
    template="source: {source}\nsource_type: {source_type}\nhas_code: {has_code}\n\n{page_content}"
)


def combine_documents(
    docs,
    document_prompt=DEFAULT_DOCUMENT_PROMPT,
    document_separator="\n\n---\n\n",
):
    cleaned_docs = [clean_document_content(doc) for doc in docs]
    doc_strings = [
        format_document(doc, document_prompt) for doc in cleaned_docs
    ]
    return document_separator.join(doc_strings)


def process_input_for_retrieval(retrieval_input):
    if isinstance(retrieval_input, list):
        retrieval_input = "\n".join(retrieval_input)
    elif isinstance(retrieval_input, dict):
        retrieval_input = json.dumps(retrieval_input)
    elif not isinstance(retrieval_input, str):
        retrieval_input = str(retrieval_input)
    return retrieval_input


def load_simple_retrieval_chain(retriever, input_key):
    default_input_chain = (
        itemgetter("standalone_question")
        | RunnablePassthrough()
        | process_input_for_retrieval
        | RunnableParallel(context=retriever)
        | itemgetter("context")
    )

    input_chain = (
        itemgetter(input_key)
        | RunnablePassthrough()
        | process_input_for_retrieval
        | RunnableParallel(context=retriever)
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


def load_cohere_rerank_chain(top_k=5):
    def load_rerank_chain(language):
        if language == "en":
            cohere_rerank = CohereRerank(
                top_n=top_k, model="rerank-english-v2.0"
            )
        else:
            cohere_rerank = CohereRerank(
                top_n=top_k, model="rerank-multilingual-v2.0"
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


def get_web_contexts(web_results):
    output_documents = []
    if not web_results:
        return []
    web_answer = web_results["web_answer"]
    # if web_answer:
    # output_documents += [
    #     Document(
    #         page_content=web_answer,
    #         metadata={
    #             "source": "you.com",
    #             "source_type": "web_answer",
    #             "has_code": None,
    #         },
    #     )
    # ]
    return (
        output_documents
        + [
            Document(
                page_content=document["context"], metadata=document["metadata"]
            )
            for document in web_results["web_context"]
        ]
        if web_results.get("web_context")
        else []
    )


def load_fusion_retriever_chain(base_retriever, embeddings, top_k=5):
    query_retrieval_chain = load_simple_retrieval_chain(
        base_retriever, "question"
    )
    standalone_query_retrieval_chain = load_simple_retrieval_chain(
        base_retriever, "standalone_question"
    )
    keywords_retrieval_chain = load_simple_retrieval_chain(
        base_retriever, "keywords"
    )
    vector_search_retrieval_chain = load_simple_retrieval_chain(
        base_retriever, "vector_search"
    )

    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)

    combined_retrieval_chain = (
        RunnableParallel(
            question=query_retrieval_chain,
            standalone_question=standalone_query_retrieval_chain,
            keywords=keywords_retrieval_chain,
            vector_search=vector_search_retrieval_chain,
            web_context=RunnableLambda(
                lambda x: get_web_contexts(x["web_results"])
            ),
        )
        | itemgetter(
            "question",
            "standalone_question",
            "keywords",
            "vector_search",
            "web_context",
        )
        | reciprocal_rank_fusion
        | redundant_filter.transform_documents
    )

    cohere_rerank_chain = load_cohere_rerank_chain(top_k=top_k)

    ranked_retrieval_chain = (
        RunnableParallel(
            context=combined_retrieval_chain,
            question=itemgetter("question"),
            language=itemgetter("language"),
        )
        | cohere_rerank_chain
    )
    return ranked_retrieval_chain
