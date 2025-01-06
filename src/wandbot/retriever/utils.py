from typing import Any, Callable, List, Sequence, Union
import numpy as np
from langchain_core.documents import BaseDocumentTransformer, Document
from pydantic import BaseModel, ConfigDict
from typing_extensions import TypeAlias
import weave

Matrix: TypeAlias = Union[List[List[float]], List[np.ndarray], np.ndarray]

def cosine_similarity(X: Matrix, Y: Matrix) -> np.ndarray:
    """Row-wise cosine similarity between two equal-width matrices."""
    if len(X) == 0 or len(Y) == 0:
        return np.array([])

    X = np.array(X)
    Y = np.array(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Number of columns in X and Y must be the same. X has shape {X.shape} "
            f"and Y has shape {Y.shape}."
        )

    X_norm = np.linalg.norm(X, axis=1)
    Y_norm = np.linalg.norm(Y, axis=1)
    # Ignore divide by zero errors run time warnings as those are handled below.
    with np.errstate(divide="ignore", invalid="ignore"):
        similarity = np.dot(X, Y.T) / np.outer(X_norm, Y_norm)
    similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
    return similarity


@weave.op
def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    embedding_list: list,
    lambda_mult: float = 0.5,
    top_k: int = 4,
) -> List[int]:
    """Calculate maximal marginal relevance.
    
    Args:
        query_embedding: query embedding
        embedding_list: list of embeddings to consider
        lambda_mult: lambda parameter (0 for MMR, 1 for standard similarity)
        top_k: number of documents to return

    Returns:
        List of indices of selected embeddings
    """
    # Handle empty case
    if min(top_k, len(embedding_list)) <= 0:
        return []
    
    # Convert embeddings to numpy array
    embedding_list = np.array(embedding_list)
    
    # Ensure query is 2D
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
    
    # Calculate similarity to query
    similarity_to_query = cosine_similarity(query_embedding, embedding_list)[0]
    
    # First selection is most similar to query
    most_similar = int(np.argmax(similarity_to_query))
    idxs = [most_similar]
    selected = np.array([embedding_list[most_similar]])
    
    # Iteratively select most diverse documents
    while len(idxs) < min(top_k, len(embedding_list)):
        best_score = -np.inf
        idx_to_add = -1
        
        # Calculate similarity to already selected docs
        similarity_to_selected = cosine_similarity(embedding_list, selected)
        
        # Calculate MMR score for each remaining document
        for i, query_score in enumerate(similarity_to_query):
            if i in idxs:
                continue
            
            # Calculate diversity penalty
            redundant_score = max(similarity_to_selected[i])
            
            # Calculate MMR score
            equation_score = lambda_mult * query_score - (1 - lambda_mult) * redundant_score
            
            if equation_score > best_score:
                best_score = equation_score
                idx_to_add = i
        
        idxs.append(idx_to_add)
        selected = np.append(selected, [embedding_list[idx_to_add]], axis=0)
    
    return idxs

@weave.op
def _filter_similar_embeddings(
    embedded_documents: Matrix,
    similarity_fn: Callable[[Matrix, Matrix], np.ndarray],
    threshold: float
) -> List[int]:
    """Filter redundant documents based on the similarity of their embeddings."""

    similarity = np.tril(similarity_fn(embedded_documents, embedded_documents), k=-1)
    redundant = np.where(similarity > threshold)
    redundant_stacked = np.column_stack(redundant)
    redundant_sorted = np.argsort(similarity[redundant])[::-1]
    included_idxs = set(range(len(embedded_documents)))
    
    for first_idx, second_idx in redundant_stacked[redundant_sorted]:
        if first_idx in included_idxs and second_idx in included_idxs:
            # Default to dropping the second document of any highly similar pair
            included_idxs.remove(second_idx)
    
    return list(sorted(included_idxs))


class EmbeddingsRedundantFilter(BaseDocumentTransformer, BaseModel):
    """Filter that drops redundant documents by comparing their embeddings."""
    
    embedding_function: Any
    """Embeddings to use for embedding document contents."""
    
    similarity_fn: Callable[[Matrix, Matrix], np.ndarray] = cosine_similarity
    """Similarity function for comparing documents. Function expected to take as input
    two matrices (List[List[float]] or numpy arrays) and return a matrix of scores 
    where higher values indicate greater similarity."""
    
    redundant_similarity_threshold: float = 0.95
    """Threshold for determining when two documents are similar enough to be considered redundant."""
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @weave.op
    def transform_documents(
        self,
        documents: Sequence[Document],
        **kwargs: Any
    ) -> Sequence[Document]:
        """Filter down documents by removing redundant ones based on embedding similarity."""
        if not documents:
            return []
            
        embedded_documents = self.embedding_function.embed(
            [doc.page_content for doc in documents]
        )
        
        # Filter similar documents
        included_idxs = _filter_similar_embeddings(
            embedded_documents,
            self.similarity_fn,
            self.redundant_similarity_threshold
        )
        
        return [documents[i] for i in sorted(included_idxs)]