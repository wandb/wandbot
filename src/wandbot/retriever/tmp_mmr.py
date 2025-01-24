# DEBUG
from typing import List, Optional, Dict, Any, Union, Tuple
from langchain_core.documents import Document
import numpy as np
import weave


def _results_to_docs(results: Any) -> List[Document]:
    # print("Running _results_to_docs")
    return [doc for doc, _ in _results_to_docs_and_scores(results)]


def _results_to_docs_and_scores(results: Any) -> List[Tuple[Document, float]]:
    # print("Running _results_to_docs_and_scores")
    return [
        # TODO: Chroma can do batch querying,
        # we shouldn't hard code to the 1st result
        (
            Document(page_content=result[0], metadata=result[1] or {}),
            result[3],
        )
        for result in zip(
            results["documents"][0],
            results["metadatas"][0],
            # results["ids"][0],
            results["distances"][0],
        )
    ]


Matrix = Union[List[List[float]], List[np.ndarray], np.ndarray]

# def debug_cosine_similarity(X: Matrix, Y: Matrix) -> np.ndarray:
#     """Row-wise cosine similarity between two equal-width matrices.

#     Raises:
#         ValueError: If the number of columns in X and Y are not the same.
#     """
#     print("Running debug_cosine_similarity")
#     if len(X) == 0 or len(Y) == 0:
#         print("Returning empty array")
#         return np.array([])

#     X = np.array(X)
#     Y = np.array(Y)
#     # Ensure Y is 2D
#     if Y.ndim == 1:
#         Y = np.expand_dims(Y, axis=0)
#     print("Finished array conversion, checking shape")
#     print(f"X.shape: {X.shape}, Y.shape: {Y.shape}")
#     if X.shape[1] != Y.shape[1]:
#         raise ValueError(
#             "Number of columns in X and Y must be the same. X has shape"
#             f"{X.shape} "
#             f"and Y has shape {Y.shape}."
#         )
#     print("Finished shape check")
#     X_norm = np.linalg.norm(X, axis=1)
#     Y_norm = np.linalg.norm(Y, axis=1)
#     # Ignore divide by zero errors run time warnings as those are handled below.
#     with np.errstate(divide="ignore", invalid="ignore"):
#         similarity = np.dot(X, Y.T) / np.outer(X_norm, Y_norm)
#     print("Finished dot product")
#     similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
#     return similarity

import logging

def debug_cosine_similarity(X: Matrix, Y: Matrix) -> np.ndarray:
    """Row-wise cosine similarity between two equal-width matrices.

    Raises:
        ValueError: If the number of columns in X and Y are not the same.
    """
    if len(X) == 0 or len(Y) == 0:
        logging.info("COSINE SIMILARITY: Returning empty array")
        return np.array([])

    X = np.array(X)
    Y = np.array(Y)
    # print(f"Checking dim_1 shape, X.shape: {X.shape}, Y.shape: {Y.shape}")
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            "Number of columns in X and Y must be the same. X has shape"
            f"{X.shape} "
            f"and Y has shape {Y.shape}."
        )

    X_norm = np.linalg.norm(X, axis=1)
    Y_norm = np.linalg.norm(Y, axis=1)
    # Ignore divide by zero errors run time warnings as those are handled below.
    with np.errstate(divide="ignore", invalid="ignore"):
        similarity = np.dot(X, Y.T) / np.outer(X_norm, Y_norm)
    similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
    return similarity


def print_rank_comparison_matrix(lc_mmr_ranks, baseline_ranks, ids, matrix_size=15):
    """
    Print ASCII version of rank comparison matrix.
    
    Args:
        lc_mmr_ranks: List of ranks in LC MMR results
        baseline_ranks: List of ranks in baseline results
        ids: List of truncated IDs to display
        matrix_size: Size of the square matrix (default 15)
    """
    # Create header
    print("\nRank Comparison Matrix (Baseline vs MMR)")
    print("   " + "".join(f"{i:4}" for i in range(matrix_size)))  # MMR ranks header
    print("   " + "─" * (4 * matrix_size))  # Top border
    
    # Create matrix rows
    for baseline_rank in range(matrix_size):
        row = f"{baseline_rank:2} │"
        for mmr_rank in range(matrix_size):
            # Check if this position has a match
            match_idx = None
            for idx, (lc, base) in enumerate(zip(lc_mmr_ranks, baseline_ranks)):
                if lc == mmr_rank and base == baseline_rank:
                    match_idx = idx
                    break
            
            # Print cell content
            if match_idx is not None:
                row += f" {ids[match_idx]:<3}"  # ID for matching ranks
            elif mmr_rank == baseline_rank:
                row += " ·  "  # Diagonal marker
            else:
                row += "    "  # Empty cell
        
        print(row)
    
    # Print statistics
    print("\nStatistics:")
    same_rank = sum(1 for lc, base in zip(lc_mmr_ranks, baseline_ranks) if lc == base)
    print(f"Same rank: {same_rank}/{len(baseline_ranks)}")
    print(f"Total matches: {len(baseline_ranks)}")
    return same_rank/len(baseline_ranks) > 0.5


@weave.op
def debug_maximal_marginal_relevance(
    query_embedding: np.ndarray,
    embedding_list: list,
    lambda_mult: float = 0.5,
    k: int = 4,
) -> List[int]:
    """Calculate maximal marginal relevance.

    Args:
        query_embedding: Query embedding.
        embedding_list: List of embeddings to select from.
        lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity.
                Defaults to 0.5.
        k: Number of Documents to return. Defaults to 4.

    Returns:
        List of indices of embeddings selected by maximal marginal relevance.
    """
    # print('Calculating MMR')
    if min(k, len(embedding_list)) <= 0:
        return []
    # print(f"len embedding_list: {len(embedding_list)}")
    # print("Running similarity_to_query")
    similarity_to_query = debug_cosine_similarity(query_embedding, embedding_list)[0]
    # print("Finished similarity_to_query")
    most_similar = int(np.argmax(similarity_to_query))
    idxs = [most_similar]
    selected = np.array([embedding_list[most_similar]])
    # print("Running while loop")
    
    while len(idxs) < min(k, len(embedding_list)):
        best_score = -np.inf
        idx_to_add = -1
        similarity_to_selected = debug_cosine_similarity(embedding_list, selected)
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
def debug_max_marginal_relevance_search_by_vector(
    results: Any,
    embedding: List[float],
    k: int = 15,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    filter: Optional[Dict[str, str]] = None,
    where_document: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> List[Document]:
    """Return docs selected using the maximal marginal relevance.

    Maximal marginal relevance optimizes for similarity to query AND diversity
    among selected documents.

    Args:
        embedding: Embedding to look up documents similar to.
        k: Number of Documents to return. Defaults to 4.
        fetch_k: Number of Documents to fetch to pass to MMR algorithm. Defaults to
            20.
        lambda_mult: Number between 0 and 1 that determines the degree
            of diversity among the results with 0 corresponding
            to maximum diversity and 1 to minimum diversity.
            Defaults to 0.5.
        filter: Filter by metadata. Defaults to None.
        where_document: dict used to filter by the documents.
                E.g. {$contains: {"text": "hello"}}.
        kwargs: Additional keyword arguments to pass to Chroma collection query.

    Returns:
        List of Documents selected by maximal marginal relevance.
    """
    # results = self.__query_collection(
    #     query_embeddings=embedding,
    #     n_results=fetch_k,
    #     where=filter,
    #     where_document=where_document,
    #     include=["metadatas", "documents", "distances", "embeddings"],
    #     **kwargs,
    # )
    
    # results = lc_retriever.vectorstore._chroma_collection.query(
    #     query_embeddings=embedding,
    #     n_results=20,
    #     include=["metadatas", "documents", "distances", "embeddings"],
    #     )
    # results["embeddings"][0]
    # print("Running max_marginal_relevance_search_by_vector")
    query_embedding = np.array(embedding, dtype=np.float32)
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
    if np.array(results["embeddings"]).shape[0] == 1 and len(np.array(results["embeddings"]).shape) == 3:
        results["embeddings"] = results["embeddings"][0]
    mmr_selected = debug_maximal_marginal_relevance(
        query_embedding,
        results["embeddings"],
        k=k,
        lambda_mult=lambda_mult,
    )
    # print(f"MRR selected: {mmr_selected}")
    # print("*"*100)

    # candidates = _results_to_docs(results)
    # print(f"results: {results}")
    # print(f"len results['documents']: {len(results['documents'])}")
    # print(f"len results['documents'][0]: {len(results['documents'][0])}")
    # print(f" first doc: {results['documents'][0]}")
    # print(f" first meta: {results['metadatas'][0]}")
    # print(f" first dist: {results['distances'][0]}")
    candidates = [Document(page_content=doc, metadata=meta, distance=dist) for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0])]
    # print(f"candidates generated, len candidates: {len(candidates)}")
    selected_results = [r for i, r in enumerate(candidates) if i in mmr_selected]
    # print(f"selected_results generated, len selected_results: {len(selected_results)}")
    return selected_results

@weave.op
async def debug_run_mmr_batch(query_embed, doc_embed, docs, metadatas, distances, top_k, lambda_mult):
    # print(f"Running debug_run_mmr_batch, {len(docs)} received")
    retrieved_results = {
        "documents": docs,
        "metadatas": metadatas,
        "distances": distances,
        "embeddings": doc_embed,
    }

    from wandbot.configs.vector_store_config import VectorStoreConfig
    from wandbot.configs.chat_config import ChatConfig
    from wandbot.retriever.base import VectorStore

    vector_store_config = VectorStoreConfig()
    chat_config = ChatConfig()
    chat_config.fetch_k = 60
    chat_config.top_k = 15
    chat_config.top_k_per_query = chat_config.top_k
    chat_config.search_type = "mmr"

    vector_store = VectorStore.from_config(
        vector_store_config=vector_store_config,
        chat_config=chat_config
    )
    retrieved_results_v1_3 = vector_store.chroma_vectorstore.collection.query(
        query_embeddings=query_embed,
        n_results=20,
        include=['documents', 'metadatas', 'distances', 'embeddings']
    )
    # print("Unranked fetch_k ids:")
    # for doc in retrieved_results_v1_3["metadatas"][0]:
    #     print(doc['id'])
    # print("*"*100)

    # RUN low-level MMR with fetch_k=20
    hybrid_lc_mmr_results = debug_max_marginal_relevance_search_by_vector(
        retrieved_results_v1_3, 
        query_embed,
        k=15, 
        fetch_k=20,
        lambda_mult=0.5
    )
    # print(f"len(hybrid_lc_mmr_results): {len(hybrid_lc_mmr_results)}")
    ids_ls = []
    for doc in hybrid_lc_mmr_results:
        # print(doc.metadata['id'])
        ids_ls.append(doc.metadata['id'])

    baseline_ids = [
        "243e3f0213a72bd6f1c19f7804e0cd30",
        "01632934ac4239d9456e16114fbede47",
        "764dfbb35e63af23d6066bd5446392f8",
        "30c474fcd2caae397c800a79d78fa7e4",
        "f0149a3b739c011fb5897583f448d4b1",
        "c4cda5b640590439c71e56e55262e7d4",
        "f89942fa78f8f991e9f1943284b044a3",
        "c4fc49f0a5a141e25280fab04e1c9e5d",
        "5f92f7b2fde95fda473abcf6d8567529",
        "2a401a62bbd374b2df3e27d5f662c484",
        "68d6263a40c88958327cc37428b15612",
        "a38b47acc51c43fa211cd958c3b64c17",
        "06868d227934977e02b20ca1c448f739",
        "246362541c79cf71af209bf3c2768eb0",
        "38b4b83bb8252e8cd594d5886a77970d"
    ]

    # Extract rankings data comparing LC MMR vs baseline
    lc_mmr_ranks = []
    baseline_ranks = []
    ids = []

    # Extract and match rankings
    for i, doc in enumerate(hybrid_lc_mmr_results):
        doc_id = doc.metadata['id']
        if doc_id in baseline_ids:  # Check if ID exists in baseline results
            lc_mmr_ranks.append(i)
            baseline_ranks.append(baseline_ids.index(doc_id))
            ids.append(doc_id[:8])

    # Create comparison matrix with fixed size 15x15
    comparison = np.zeros((15, 15))
    for i in range(15):
        comparison[i, i] = 0.2
    for lc, base, id_ in zip(lc_mmr_ranks, baseline_ranks, ids):
        comparison[lc, base] = 1

    # Update statistics calculation
    same_rank = sum(1 for lc, base in zip(lc_mmr_ranks, baseline_ranks) if lc == base)
    overlapping = len(lc_mmr_ranks)

    # Get unique IDs from LC MMR results
    lc_ids = {doc.metadata['id'] for doc in hybrid_lc_mmr_results}
    missing_from_baseline = len(lc_ids - set(baseline_ids))
    missing_baseline_ids = set(baseline_ids) - lc_ids

    # print("\nRetrieval Statistics:")
    # print(f"Same rank in both versions: {same_rank}/{len(baseline_ids)}")
    # print(f"Overlapping IDs (any rank): {overlapping}/{len(baseline_ids)}")
    # print(f"IDs in Hybrid MMR but not in baseline: {missing_from_baseline}")
    # print("\nBaseline IDs missing from Hybrid MMR results:")
    # for missing_id in missing_baseline_ids:
    #     print(f"- {missing_id}")
    # print("*"*100)
    
    # same_rank_pct = print_rank_comparison_matrix(lc_mmr_ranks, baseline_ranks, ids)
    # if same_rank_pct > 0.5:
    #     raise ValueError("Same rank percentage looking good")

    # verify 

    # hybrid_lc_mmr_results = debug_max_marginal_relevance_search_by_vector(
    #     results=retrieved_results, 
    #     embedding=query_embed,
    #     k=15, 
    #     fetch_k=20,
    #     lambda_mult=0.5
    # )   
    # print("returning from debug_run_mmr_batch")
    return hybrid_lc_mmr_results