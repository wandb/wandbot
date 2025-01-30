import os
import asyncio
import weave
from typing import List, Union, Tuple, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential
import base64
import numpy as np
import traceback
import sys

from wandbot.configs.vector_store_config import VectorStoreConfig
from wandbot.utils import get_logger, ErrorInfo, get_error_file_path
from wandbot.schema.api_status import APIStatus

logger = get_logger(__name__)
vector_store_config = VectorStoreConfig()

# Valid input types for Cohere embeddings
VALID_COHERE_INPUT_TYPES = ["search_document", "search_query", "classification", "clustering", "image"]

class BaseEmbeddingModel:
    COMPONENT_NAME = "embedding"  # Default component name
    
    def __init__(self, 
                 model_name: str,
                 n_parallel_api_calls: int = 50,
                 max_retries: int = 3,
                 timeout: int = 30,
                 **kwargs):
        self.model_name = model_name
        self.n_parallel_api_calls = n_parallel_api_calls
        self.max_retries = max_retries
        self.timeout = timeout

    @weave.op
    def embed(self, input: Union[str, List[str]]) -> Tuple[List[List[float]], APIStatus]:
        raise NotImplementedError("Subclasses must implement embed method")

    def __call__(self, input: Union[str, List[str]] = None) -> List[List[float]]:
        embeddings, api_status = self.embed(input)
        if not api_status.success:
            raise RuntimeError(api_status.error_info.error_message)
        return embeddings

class OpenAIEmbeddingModel(BaseEmbeddingModel):
    COMPONENT_NAME = "openai_embedding"
    
    def __init__(self, model_name: str, dimensions: int, encoding_format: str = "float", **kwargs):
        super().__init__(model_name, **kwargs)
        self.dimensions = dimensions
        self.encoding_format = encoding_format
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=self.max_retries, timeout=self.timeout)

    async def _run_openai_embeddings(self, inputs: List[str]) -> Tuple[List[List[float]], APIStatus]:
        api_status = APIStatus(component=self.COMPONENT_NAME, success=True)
        try:
            semaphore = asyncio.Semaphore(self.n_parallel_api_calls)
            async def get_single_openai_embedding(text):
                async with semaphore:
                    response = await self.client.embeddings.create(
                        input=text, model=self.model_name,
                        encoding_format=self.encoding_format,
                        dimensions=self.dimensions)
                    embedding = response.data[0].embedding
                    if self.encoding_format == "base64" and isinstance(embedding, str):
                        decoded_embeddings = base64.b64decode(embedding)
                        return np.frombuffer(decoded_embeddings, dtype=np.float32).tolist()
                    return embedding
            embeddings = await asyncio.gather(*[get_single_openai_embedding(text) for text in inputs])
            return embeddings, api_status
        except Exception as e:
            error_info = ErrorInfo(
                component=self.COMPONENT_NAME,
                has_error=True,
                error_message=str(e),
                error_type=type(e).__name__,
                stacktrace=''.join(traceback.format_exc()),
                file_path=get_error_file_path(sys.exc_info()[2])
            )
            return None, APIStatus(
                component=self.COMPONENT_NAME,
                success=False,
                error_info=error_info
            )

    def embed(self, input: Union[str, List[str]]) -> Tuple[List[List[float]], APIStatus]:
        inputs = [input] if isinstance(input, str) else input
        try:
            return asyncio.run(self._run_openai_embeddings(inputs))
        except Exception as e:
            logger.error(f"EMBEDDING: Error calling OpenAI embed:\n{e}")
            error_info = ErrorInfo(
                has_error=True,
                error_message=str(e),
                error_type=type(e).__name__,
                stacktrace=''.join(traceback.format_exc()),
                file_path=get_error_file_path(sys.exc_info()[2]),
                component=self.COMPONENT_NAME
            )
            return None, APIStatus(
                component=self.COMPONENT_NAME,
                success=False,
                error_info=error_info
            )

class CohereEmbeddingModel(BaseEmbeddingModel):
    COMPONENT_NAME = "cohere_embedding"
    
    def __init__(self, model_name: str, input_type: str, dimensions: int = None, **kwargs):
        super().__init__(model_name, **kwargs)
        if not input_type:
            raise ValueError("input_type must be specified for Cohere embeddings")
        if input_type not in VALID_COHERE_INPUT_TYPES:
            raise ValueError(f"Invalid input_type: {input_type}. Must be one of {VALID_COHERE_INPUT_TYPES}")
        self.input_type = input_type
        self.dimensions = dimensions
        try:
            import cohere
            from cohere.core.request_options import RequestOptions
            self.client = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
            self.RequestOptions = RequestOptions  # Store the class for later use
        except Exception as e:
            logger.error(f'Unable to initialise Cohere client:\n{e}')
            raise

    def embed(self, input: Union[str, List[str]]) -> Tuple[List[List[float]], APIStatus]:
        api_status = APIStatus(component=self.COMPONENT_NAME, success=True)
        inputs = [input] if isinstance(input, str) else input
        try:
            response = self.client.embed(
                texts=inputs,
                model=self.model_name,
                input_type=self.input_type,
                embedding_types=["float"],
                request_options=self.RequestOptions(max_retries=self.max_retries, timeout_in_seconds=self.timeout)
            )
            embeddings = response.embeddings.float
            # Convert to numpy arrays
            if isinstance(embeddings, list):
                if len(embeddings) == 1:
                    return np.array(embeddings[0]), api_status
                return [np.array(emb) for emb in embeddings], api_status
            return np.array(embeddings), api_status
        except Exception as e:
            logger.error(f"EMBEDDING: Error calling Cohere embed:\n{e}")
            error_info = ErrorInfo(
                has_error=True,
                error_message=str(e),
                error_type=type(e).__name__,
                stacktrace=''.join(traceback.format_exc()),
                file_path=get_error_file_path(sys.exc_info()[2]),
                component=self.COMPONENT_NAME
            )
            return None, APIStatus(
                component=self.COMPONENT_NAME,
                success=False,
                error_info=error_info
            )

class EmbeddingModel:
    PROVIDER_MAP = {
        "openai": OpenAIEmbeddingModel,
        "cohere": CohereEmbeddingModel
    }

    def __init__(self, provider: str, **kwargs):
        provider = provider.lower()
        if provider not in self.PROVIDER_MAP:
            raise ValueError(f"Unsupported provider: {provider}. Choose from {list(self.PROVIDER_MAP.keys())}")
        
        if provider == "openai" and "dimensions" not in kwargs:
            raise ValueError("`dimensions` needs to be specified when using OpenAI embeddings models")
        
        if provider == "cohere" and "input_type" not in kwargs:
            raise ValueError("input_type must be specified for Cohere embeddings")
        
        self.model = self.PROVIDER_MAP[provider](**kwargs)

    def embed(self, input: Union[str, List[str]] = None) -> Tuple[List[List[float]], APIStatus]:
        try:
            embeddings, api_status = self.model.embed(input)
            return embeddings, api_status
        except Exception as e:
            logger.error(f"EMBEDDING: Error in embedding model: {e}")
            error_info = ErrorInfo(
                has_error=True,
                error_message=str(e),
                error_type=type(e).__name__,
                stacktrace=''.join(traceback.format_exc()),
                file_path=get_error_file_path(sys.exc_info()[2]),
                component=self.model.COMPONENT_NAME  # Use provider's component name
            )
            return None, APIStatus(
                component=self.model.COMPONENT_NAME,  # Use provider's component name
                success=False,
                error_info=error_info
            )

    def __call__(self, input: Union[str, List[str]] = None) -> List[List[float]]:
        """Required interface for Chroma's EmbeddingFunction"""
        embeddings, api_status = self.embed(input)
        if not api_status.success:
            raise RuntimeError(api_status.error_info.error_message)  # Raise for Chroma to handle
        return embeddings