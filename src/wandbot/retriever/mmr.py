from typing import List, Any
import numpy as np
import weave

from wandbot.retriever.utils import cosine_similarity
from wandbot.schema.document import Document
@weave.op
def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    embedding_list: list,
    top_k: int,
    lambda_mult: float,
) -> List[int]:
    """Calculate maximal marginal relevance.

    Args:
        query_embedding: Query embedding.
        embedding_list: List of embeddings to select from.
        lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity.
        k: Number of Documents to return.

    Returns:
        List of indices of embeddings selected by maximal marginal relevance.
    """
    if min(top_k, len(embedding_list)) <= 0:
        return []
    similarity_to_query = cosine_similarity(query_embedding, embedding_list)[0]
    most_similar = int(np.argmax(similarity_to_query))
    idxs = [most_similar]
    selected = np.array([embedding_list[most_similar]])
    
    while len(idxs) < min(top_k, len(embedding_list)):
        best_score = -np.inf
        idx_to_add = -1
        similarity_to_selected = cosine_similarity(embedding_list, selected)
        for i, query_score in enumerate(similarity_to_query):
            if i in idxs:
                continue
            redundant_score = max(similarity_to_selected[i])
            equation_score = (
                lambda_mult * query_score - (1 - lambda_mult) * redundant_score
            )
            if equation_score > best_score:
                best_score = equation_score
                idx_to_add = i
        idxs.append(idx_to_add)
        selected = np.append(selected, [embedding_list[idx_to_add]], axis=0)
    return idxs

@weave.op
def max_marginal_relevance_search_by_vector(
    retrieved_results: Any,
    embedding: List[float],
    top_k: int,
    lambda_mult: float,
) -> List[Document]:
    """Return docs selected using the maximal marginal relevance.

    Maximal marginal relevance optimizes for similarity to query AND diversity
    among selected documents.

    Args:
        retrieved_results: Retrieved results from vector store.
        embedding: Embedding to look up documents similar to.
        top_k: Number of Documents to return.
        lambda_mult: Number between 0 and 1 that determines the degree
            of diversity among the results with 0 corresponding
            to maximum diversity and 1 to minimum diversity.

    Returns:
        List of Documents selected by maximal marginal relevance.
    """
    query_embedding = np.array(embedding, dtype=np.float32)
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)

    if np.array(retrieved_results["embeddings"]).shape[0] == 1 and len(np.array(retrieved_results["embeddings"]).shape) == 3:
        retrieved_results["embeddings"] = retrieved_results["embeddings"][0]
    
    mmr_selected = maximal_marginal_relevance(
        query_embedding,
        retrieved_results["embeddings"],
        top_k=top_k,
        lambda_mult=lambda_mult,
    )

    candidates = [Document(page_content=doc, metadata=meta, distance=dist) for doc, meta, dist in zip(
        retrieved_results["documents"], 
        retrieved_results["metadatas"], 
        retrieved_results["distances"])]
    selected_results = [r for i, r in enumerate(candidates) if i in mmr_selected]
    return selected_results