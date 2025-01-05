import os
import asyncio
import weave
import warnings
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential

class EmbeddingModel():
    def __init__(self, 
                    provider:str = "openai", 
                    model_name:str = "text-embedding-3-small",
                    input_type:str = None,
                    dimensions:int=512,
                    n_parallel_api_calls:int = 50,
                ):
        self.provider = provider.lower()
        
        if self.provider == "openai":
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.provider == "cohere":
            import cohere
            self.client = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))

        self.model_name = model_name
        if self.provider == "cohere" and self.input_type is None:
            warnings.warn()
            raise ValueError("input_type must be specified for Cohere embeddings, set `input_type` to \
 either 'search_query' or 'search_document' depending on whether it is the query or document being embedded.")
        self.input_type = input_type 
        self.dimensions = dimensions 
        self.n_parallel_api_calls = n_parallel_api_calls
        self._loop = None

    async def _cleanup(self):
        """Cleanup method to properly close the client"""
        if self.provider == "openai":
            await self.client.close()

    @weave.op
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=120),
        reraise=True
    )
    def embed(self, inputs: List[str]) -> List[List[float]]:
        """Takes a list of texts and output a list of embeddings lists"""
        if self.provider == "openai":
            async def run_embeddings():
                try:
                    semaphore = asyncio.Semaphore(self.n_parallel_api_calls)
                    async def get_single_embedding(text):
                        async with semaphore:
                            response = await self.client.embeddings.create(
                                input=text,
                                model=self.model_name,
                                encoding_format="float",
                                dimensions=self.dimensions
                            )
                            return response.data[0].embedding

                    tasks = [get_single_embedding(text) for text in inputs]
                    results = await asyncio.gather(*tasks)
                    await self._cleanup()
                    return results
                except Exception as e:
                    await self._cleanup()
                    raise e

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(run_embeddings())
            finally:
                loop.close()
        
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