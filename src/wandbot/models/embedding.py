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

logger = get_logger(__name__)
vector_store_config = VectorStoreConfig()

class BaseEmbeddingModel:
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
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=120), reraise=True)
    def embed(self, input: Union[str, List[str]]) -> Tuple[List[List[float]], ErrorInfo]:
        raise NotImplementedError("Subclasses must implement embed method")

    def __call__(self, input: Union[str, List[str]] = None) -> List[List[float]]:
        return self.embed(input)

class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name: str, dimensions: int, encoding_format: str = "float", **kwargs):
        super().__init__(model_name, **kwargs)
        self.dimensions = dimensions
        self.encoding_format = encoding_format

    async def _run_openai_embeddings(self, inputs: List[str]) -> Tuple[List[List[float]], ErrorInfo]:
        from openai import AsyncOpenAI
        error_info = ErrorInfo(component="embedding")
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=self.max_retries, timeout=self.timeout)
        try:
            semaphore = asyncio.Semaphore(self.n_parallel_api_calls)
            async def get_single_openai_embedding(text):
                async with semaphore:
                    response = await client.embeddings.create(
                        input=text, model=self.model_name,
                        encoding_format=self.encoding_format,
                        dimensions=self.dimensions)
                    embedding = response.data[0].embedding
                    if self.encoding_format == "base64" and isinstance(embedding, str):
                        decoded_embeddings = base64.b64decode(embedding)
                        return np.frombuffer(decoded_embeddings, dtype=np.float32).tolist()
                    return embedding
            embeddings = await asyncio.gather(*[get_single_openai_embedding(text) for text in inputs])
            return embeddings, error_info
        except Exception as e:
            error_info.has_error = True
            error_info.error_message = str(e)
            error_info.error_type = type(e).__name__
            error_info.stacktrace = ''.join(traceback.format_exc())
            error_info.file_path = get_error_file_path(sys.exc_info()[2])
            return None, error_info
        finally:
            await client.close()

    def embed(self, input: Union[str, List[str]]) -> Tuple[List[List[float]], ErrorInfo]:
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
                component="embedding"
            )
            return None, error_info

class CohereEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name: str, input_type: str, **kwargs):
        super().__init__(model_name, **kwargs)
        self.input_type = input_type
        try:
            import cohere
            from cohere.core.request_options import RequestOptions
            self.client = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
            self.RequestOptions = RequestOptions  # Store the class for later use
        except Exception as e:
            logger.error(f'Unable to initialise Cohere client:\n{e}')
            raise

    def embed(self, input: Union[str, List[str]]) -> Tuple[List[List[float]], ErrorInfo]:
        error_info = ErrorInfo(component="embedding")
        inputs = [input] if isinstance(input, str) else input
        try:
            response = self.client.embed(
                texts=inputs,
                model=self.model_name,
                input_type=self.input_type,
                embedding_types=["float"],
                request_options=self.RequestOptions(max_retries=self.max_retries, timeout_in_seconds=self.timeout)
            )
            return response.embeddings.float, error_info
        except Exception as e:
            logger.error(f"EMBEDDING: Error calling Cohere embed:\n{e}")
            error_info.has_error = True
            error_info.error_message = str(e)
            error_info.error_type = type(e).__name__
            error_info.stacktrace = ''.join(traceback.format_exc())
            error_info.file_path = get_error_file_path(sys.exc_info()[2])
            return None, error_info

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

    def embed(self, input: Union[str, List[str]] = None) -> Tuple[List[List[float]], ErrorInfo]:
        try:
            embeddings, error_info = self.model.embed(input)
            return embeddings, error_info
        except Exception as e:
            logger.error(f"EMBEDDING: Error in embedding model: {e}")
            error_info = ErrorInfo(
                has_error=True,
                error_message=str(e),
                error_type=type(e).__name__,
                stacktrace=''.join(traceback.format_exc()),
                file_path=get_error_file_path(sys.exc_info()[2]),
                component="embedding"
            )
            return None, error_info

    def __call__(self, input: Union[str, List[str]] = None) -> List[List[float]]:
        """Required interface for Chroma's EmbeddingFunction"""
        embeddings, error_info = self.embed(input)  # Ignore error info for Chroma interface
        if error_info.has_error:
            raise RuntimeError(error_info.error_message)  # Raise for Chroma to handle
        return embeddings