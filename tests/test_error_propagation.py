from pathlib import Path

import pytest
from dotenv import load_dotenv

from tests.test_config import TestConfig
from wandbot.chat.rag import RAGPipeline
from wandbot.configs.chat_config import ChatConfig
from wandbot.configs.vector_store_config import VectorStoreConfig
from wandbot.models.embedding import EmbeddingModel
from wandbot.rag.retrieval import FusionRetrievalEngine
from wandbot.retriever.chroma import ChromaVectorStore
from wandbot.schema.api_status import APIStatus
from wandbot.schema.document import Document

# Load environment variables from .env in project root
ENV_PATH = Path(__file__).parent.parent / '.env'
load_dotenv(ENV_PATH, override=True)

@pytest.fixture
def chat_config():
    # Create base config with all settings
    config = ChatConfig()
    
    # Override with test retry settings
    test_config = TestConfig()
    
    # Update only retry-related settings that exist in ChatConfig
    config.llm_max_retries = test_config.llm_max_retries
    config.llm_retry_min_wait = test_config.llm_retry_min_wait
    config.llm_retry_max_wait = test_config.llm_retry_max_wait
    config.llm_retry_multiplier = test_config.llm_retry_multiplier
    
    config.embedding_max_retries = test_config.embedding_max_retries
    config.embedding_retry_min_wait = test_config.embedding_retry_min_wait
    config.embedding_retry_max_wait = test_config.embedding_retry_max_wait
    config.embedding_retry_multiplier = test_config.embedding_retry_multiplier
    
    config.reranker_max_retries = test_config.reranker_max_retries
    config.reranker_retry_min_wait = test_config.reranker_retry_min_wait
    config.reranker_retry_max_wait = test_config.reranker_retry_max_wait
    config.reranker_retry_multiplier = test_config.reranker_retry_multiplier
    
    return config

@pytest.fixture
def vector_store_config():
    return VectorStoreConfig()

@pytest.fixture
def embedding_model(vector_store_config, chat_config):
    return EmbeddingModel(
        provider=vector_store_config.embeddings_provider,
        model_name=vector_store_config.embeddings_model_name,
        dimensions=vector_store_config.embeddings_dimensions,
        encoding_format=vector_store_config.embeddings_encoding_format,
        max_retries=chat_config.embedding_max_retries
    )

@pytest.fixture
def vector_store(embedding_model, vector_store_config, chat_config):
    return ChromaVectorStore(
        embedding_function=embedding_model,
        vector_store_config=vector_store_config,
        chat_config=chat_config
    )

@pytest.fixture
def retrieval_engine(vector_store, chat_config):
    return FusionRetrievalEngine(
        vector_store=vector_store,
        chat_config=chat_config
    )

@pytest.fixture
def rag_pipeline(vector_store, chat_config):
    return RAGPipeline(
        vector_store=vector_store,
        chat_config=chat_config
    )

def test_successful_embedding(vector_store_config, chat_config):
    """Test successful embedding with API status"""
    model = EmbeddingModel(
        provider=vector_store_config.embeddings_provider,
        model_name=vector_store_config.embeddings_model_name,
        dimensions=vector_store_config.embeddings_dimensions,
        encoding_format=vector_store_config.embeddings_encoding_format,
        max_retries=chat_config.embedding_max_retries
    )
    
    embeddings, api_status = model.embed("test query")
    
    assert embeddings is not None
    assert isinstance(api_status, APIStatus)
    assert api_status.success
    assert api_status.error_info is None
    assert api_status.component == f"{vector_store_config.embeddings_provider}_embedding"

def test_invalid_embedding_model(vector_store_config, chat_config):
    """Test error propagation with invalid embedding model"""
    model = EmbeddingModel(
        provider=vector_store_config.embeddings_provider,
        model_name="invalid-model",
        dimensions=vector_store_config.embeddings_dimensions,
        encoding_format=vector_store_config.embeddings_encoding_format,
        max_retries=chat_config.embedding_max_retries
    )
    
    embeddings, api_status = model.embed("test query")
    
    assert embeddings is None
    assert isinstance(api_status, APIStatus)
    assert not api_status.success
    assert api_status.error_info is not None
    assert "does not exist" in api_status.error_info.error_message.lower()
    assert api_status.component == f"{vector_store_config.embeddings_provider}_embedding"
    assert api_status.error_info.stacktrace is not None
    assert api_status.error_info.file_path is not None

@pytest.mark.asyncio
async def test_successful_reranking(retrieval_engine, chat_config):
    """Test successful reranking with API status"""
    docs = [
        Document(page_content="test doc 1", metadata={"id": "1"}),
        Document(page_content="test doc 2", metadata={"id": "2"})
    ]
    
    reranked_docs, api_status = await retrieval_engine._async_rerank_results(
        query="test query",
        context=docs,
        top_k=chat_config.top_k,
        language="en"
    )
    
    assert len(reranked_docs) == 2
    assert isinstance(api_status, APIStatus)
    assert api_status.success
    assert api_status.error_info is None
    assert api_status.component == "reranker_api"

@pytest.mark.asyncio
async def test_invalid_reranker_model(retrieval_engine, chat_config):
    """Test error propagation with invalid reranker model"""
    # Save original model name
    original_model = chat_config.english_reranker_model
    chat_config.english_reranker_model = "invalid-model"
    
    docs = [
        Document(page_content="test doc 1", metadata={"id": "1"}),
        Document(page_content="test doc 2", metadata={"id": "2"})
    ]
    
    reranked_docs, api_status = await retrieval_engine._async_rerank_results(
        query="test query",
        context=docs,
        top_k=chat_config.top_k,
        language="en"
    )
    
    # Restore original model name
    chat_config.english_reranker_model = original_model
    
    assert len(reranked_docs) == 0  # Empty list returned on error
    assert isinstance(api_status, APIStatus)
    assert not api_status.success
    assert api_status.error_info is not None
    assert api_status.component == "reranker_api"
    assert api_status.error_info.stacktrace is not None
    assert api_status.error_info.file_path is not None

def test_error_propagation_in_retrieval(retrieval_engine, chat_config):
    """Test error propagation through the retrieval pipeline using MMR search"""
    # Save original model name
    original_model = chat_config.english_reranker_model
    chat_config.english_reranker_model = "invalid-model"
    
    inputs = {
        "standalone_query": "test query",
        "all_queries": ["test query"],
        "language": "en"
    }
    
    results = retrieval_engine.vectorstore.max_marginal_relevance_search(
        query_texts=inputs["all_queries"],
        top_k=chat_config.top_k,
        fetch_k=chat_config.fetch_k,
        lambda_mult=chat_config.mmr_lambda_mult
    )
    
    # Restore original model name
    chat_config.english_reranker_model = original_model
    
    # Check API status objects are properly propagated
    assert "_embedding_status" in results
    assert isinstance(results["_embedding_status"], APIStatus)
    assert results["_embedding_status"].component == f"{retrieval_engine.vectorstore.embedding_function.model.COMPONENT_NAME}"
    
    # Check that we got results
    assert len(results) > 0

def test_error_propagation_in_similarity_search(retrieval_engine, chat_config):
    """Test error propagation through the retrieval pipeline using similarity search"""
    # Save original model name
    original_model = chat_config.english_reranker_model
    chat_config.english_reranker_model = "invalid-model"

    inputs = {
        "standalone_query": "test query",
        "all_queries": ["test query"],
        "language": "en"
    }

    results = retrieval_engine.vectorstore.similarity_search(
        query_texts=inputs["all_queries"],
        top_k=chat_config.top_k
    )

    # Restore original model name
    chat_config.english_reranker_model = original_model

    # Check that we got results for our query
    assert inputs["standalone_query"] in results
    assert len(results[inputs["standalone_query"]]) > 0

    # Check API status is properly propagated
    assert "_embedding_status" in results
    assert isinstance(results["_embedding_status"], APIStatus)
    assert results["_embedding_status"].component == f"{retrieval_engine.vectorstore.embedding_function.model.COMPONENT_NAME}" 