import pytest
import os
from dotenv import load_dotenv
import numpy as np
from pathlib import Path

from wandbot.models.embedding import (
    EmbeddingModel, 
    VALID_COHERE_INPUT_TYPES
)
from wandbot.schema.api_status import APIStatus

# Load environment variables from .env in project root
load_dotenv(Path(__file__).parent.parent / ".env")

cohere_models = [
    "embed-english-v3.0",
]

openai_models = [
    "text-embedding-3-small",
]

# Basic model creation tests
@pytest.mark.parametrize("model_name", cohere_models)
def test_cohere_embedding_creation(model_name):
    model = EmbeddingModel(
        provider="cohere",
        model_name=model_name,
        dimensions=1024,
        encoding_format="float",
        input_type=VALID_COHERE_INPUT_TYPES[0]  # Use first valid type
    )
    assert model.model.model_name == model_name
    assert model.model.dimensions == 1024
    assert model.model.client is not None  # Just verify client exists

@pytest.mark.parametrize("model_name", openai_models)
def test_openai_embedding_creation(model_name):
    model = EmbeddingModel(
        provider="openai",
        model_name=model_name,
        dimensions=1536,
        encoding_format="float"
    )
    assert model.model.model_name == model_name
    assert model.model.dimensions == 1536
    assert model.model.client.api_key == os.getenv("OPENAI_API_KEY")

# Embedding generation tests
@pytest.mark.parametrize("model_name", cohere_models)
def test_cohere_embed_single(model_name):
    model = EmbeddingModel(
        provider="cohere",
        model_name=model_name,
        dimensions=1024,
        encoding_format="float",
        input_type=VALID_COHERE_INPUT_TYPES[0]
    )
    embeddings, api_status = model.embed("This is a test sentence.")
    
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (1024,)
    assert isinstance(api_status, APIStatus)
    assert api_status.success
    assert api_status.error_info is None
    assert api_status.component == "cohere_embedding"

@pytest.mark.parametrize("model_name", cohere_models)
def test_cohere_embed_batch(model_name):
    model = EmbeddingModel(
        provider="cohere",
        model_name=model_name,
        dimensions=1024,
        encoding_format="float",
        input_type=VALID_COHERE_INPUT_TYPES[0]
    )
    texts = ["First test sentence.", "Second test sentence.", "Third test sentence."]
    embeddings, api_status = model.embed(texts)
    
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(texts)
    assert all(isinstance(emb, np.ndarray) for emb in embeddings)
    assert all(emb.shape == (1024,) for emb in embeddings)
    assert isinstance(api_status, APIStatus)
    assert api_status.success
    assert api_status.error_info is None
    assert api_status.component == "cohere_embedding"

@pytest.mark.parametrize("model_name", openai_models)
def test_openai_embed_single(model_name):
    model = EmbeddingModel(
        provider="openai",
        model_name=model_name,
        dimensions=1536,
        encoding_format="float"
    )
    embeddings, api_status = model.embed("This is a test sentence.")
    
    # Convert to numpy array if it's not already
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings).squeeze()
    
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (1536,)
    assert isinstance(api_status, APIStatus)
    assert api_status.success
    assert api_status.error_info is None
    assert api_status.component == "openai_embedding"

@pytest.mark.parametrize("model_name", openai_models)
def test_openai_embed_batch(model_name):
    model = EmbeddingModel(
        provider="openai",
        model_name=model_name,
        dimensions=1536,
        encoding_format="float"
    )
    texts = ["First test sentence.", "Second test sentence.", "Third test sentence."]
    embeddings, api_status = model.embed(texts)
    
    # Convert to numpy arrays if they're not already
    if isinstance(embeddings, list):
        embeddings = [np.array(emb).squeeze() for emb in embeddings]
    
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(texts)
    assert all(isinstance(emb, np.ndarray) for emb in embeddings)
    assert all(emb.shape == (1536,) for emb in embeddings)
    assert isinstance(api_status, APIStatus)
    assert api_status.success
    assert api_status.error_info is None
    assert api_status.component == "openai_embedding"

# Error handling tests
def test_invalid_cohere_model():
    model = EmbeddingModel(
        provider="cohere",
        model_name="invalid-model",
        dimensions=1024,
        encoding_format="float",
        input_type=VALID_COHERE_INPUT_TYPES[0]
    )
    embeddings, api_status = model.embed("Test sentence.")
    
    assert embeddings is None
    assert isinstance(api_status, APIStatus)
    assert not api_status.success
    assert api_status.error_info is not None
    # The error message could be about model invalidity or internal server error
    assert any(phrase in api_status.error_info.error_message.lower() for phrase in [
        "model", "not found", "invalid", "does not exist", "internal server error"
    ])
    assert api_status.component == "cohere_embedding"
    assert api_status.error_info.error_type is not None
    assert api_status.error_info.stacktrace is not None
    assert api_status.error_info.file_path is not None

def test_invalid_openai_model():
    model = EmbeddingModel(
        provider="openai",
        model_name="invalid-model",
        dimensions=1536,
        encoding_format="float"
    )
    embeddings, api_status = model.embed("Test sentence.")
    
    assert embeddings is None
    assert isinstance(api_status, APIStatus)
    assert not api_status.success
    assert api_status.error_info is not None
    assert "model" in api_status.error_info.error_message.lower()
    assert api_status.component == "openai_embedding"
    assert api_status.error_info.error_type is not None
    assert api_status.error_info.stacktrace is not None
    assert api_status.error_info.file_path is not None

def test_invalid_provider():
    with pytest.raises(ValueError, match="Unsupported provider"):
        EmbeddingModel(
            provider="invalid",
            model_name="some-model",
            dimensions=1024,
            encoding_format="float"
        )

def test_missing_input_type_cohere():
    with pytest.raises(ValueError, match="input_type must be specified for Cohere embeddings"):
        EmbeddingModel(
            provider="cohere",
            model_name="embed-english-v3.0",
            dimensions=1024,
            encoding_format="float"
        )

def test_invalid_input_type_cohere():
    with pytest.raises(ValueError, match="Invalid input_type"):
        EmbeddingModel(
            provider="cohere",
            model_name="embed-english-v3.0",
            dimensions=1024,
            encoding_format="float",
            input_type="invalid_type"
        ) 