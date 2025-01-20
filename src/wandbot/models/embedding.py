import os
import asyncio
import weave
import warnings
from typing import List, Union
from tenacity import retry, stop_after_attempt, wait_exponential

from wandbot.configs.vector_store_config import VectorStoreConfig
from wandbot.utils import get_logger

import base64
import numpy as np

logger = get_logger(__name__)

vector_store_config = VectorStoreConfig()

class EmbeddingModel():
    """
    Current providers: `openai` and `cohere`
    """
    def __init__(self, 
                    provider:str, 
                    model_name:str,
                    input_type:str = None,
                    dimensions:int = None,
                    n_parallel_api_calls:int = 50,
                    encoding_format:str = "float",
                ):
        self.provider = provider.lower()
        self.model_name = model_name
        
        # Set api client
        if self.provider == "openai":
            try:
                from openai import AsyncOpenAI
                self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except Exception as e:
                logger.error(f'Unable to initialise embedding model client:\n{e}\n')
                raise e
        elif self.provider == "cohere":
            try:
                import cohere
                self.client = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
            except Exception as e:
                logger.error(f'Unable to initialise embedding model client:\n{e}\n')
                raise e
            self._loop = None

        self.input_type = input_type 
        if self.provider == "cohere" and self.input_type is None:
            warnings.warn()
            raise ValueError("input_type must be specified for Cohere embeddings, set `input_type` to \
 either 'search_query' or 'search_document' depending on whether it is the query or document being embedded.")
        
        self.dimensions = dimensions 
        if self.provider == "openai" and self.dimensions is None:
            warnings.warn()
            raise ValueError("`dimensions` needs to be specified when using OpenAI embeddings models")
        
        self.n_parallel_api_calls = n_parallel_api_calls
        self.encoding_format = encoding_format
    @weave.op
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=120),
        reraise=True
    )
    def embed(self, input: Union[str, List[str]]) -> List[List[float]]:
        """Takes a list of texts and output a list of embeddings lists"""
        try:
            if isinstance(input, str):
                inputs = [input]
            else:
                inputs = input
            
            if self.provider == "openai":
                # Create a new client for each request to avoid lifecycle issues
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                
                async def run_openai_embeddings():
                    try:
                        semaphore = asyncio.Semaphore(self.n_parallel_api_calls)
                        async def get_single_openai_embedding(text):
                            async with semaphore:
                                response = await client.embeddings.create(
                                    input=text,
                                    model=self.model_name,
                                    encoding_format=self.encoding_format,
                                    dimensions=self.dimensions
                                )
                                if self.encoding_format == "base64":
                                    decoded_embeddings = base64.b64decode(response.data[0].embedding)
                                    embeddings = np.frombuffer(decoded_embeddings, dtype=np.float32).tolist()
                                    return embeddings
                                else:
                                    return response.data[0].embedding

                        return await asyncio.gather(*[get_single_openai_embedding(text) for text in inputs])
                    finally:
                        await client.close()

                # Use a new event loop for each request
                try:
                    return asyncio.run(run_openai_embeddings())
                except Exception as e:
                    raise e
            
            elif self.provider == "cohere":
                try:
                    response = self.client.embed(
                        texts=inputs,
                        model=self.model_name,
                        input_type=self.input_type,
                        embedding_types=["float"]
                    )
                    return response.embeddings.float
                except Exception as e:
                    raise e
                
        except Exception as e:
            logger.error(f"EMBEDDING: Error calling `embed`:\n{e}\n")
    
    @weave.op
    def __call__(self, input: Union[str, List[str]] = None) -> List[List[float]]:
        return self.embed(input)