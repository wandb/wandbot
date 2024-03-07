import datetime
from typing import List, Optional, Tuple

from langchain_anthropic import ChatAnthropic
from langchain_community.callbacks import get_openai_callback
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from wandbot.rag.query_handler import QueryEnhancer
from wandbot.rag.response_synthesis import ResponseSynthesizer
from wandbot.rag.retrieval import FusionRetrieval
from wandbot.retriever import VectorStore
from wandbot.utils import Timer, get_logger

logger = get_logger(__name__)


def get_stats_dict_from_token_callback(token_callback):
    return {
        "total_tokens": token_callback.total_tokens,
        "prompt_tokens": token_callback.prompt_tokens,
        "completion_tokens": token_callback.completion_tokens,
        "successful_requests": token_callback.successful_requests,
    }


def get_stats_dict_from_timer(timer):
    return {
        "start_time": timer.start,
        "end_time": timer.stop,
        "time_taken": timer.elapsed,
    }


class RAGPipelineOutput(BaseModel):
    question: str
    answer: str
    sources: str
    source_documents: str
    system_prompt: str
    model: str
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    time_taken: float
    start_time: datetime.datetime
    end_time: datetime.datetime


class RAGPipeline:
    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = 5,
        search_type: str = "mmr",
    ):
        self.vector_store = vector_store
        self.query_enhancer = QueryEnhancer()
        self.retrieval = FusionRetrieval(
            vector_store=vector_store, top_k=top_k, search_type=search_type
        )
        self.response_synthesizer = ResponseSynthesizer()

    def generate_multi_modal_initial_response(
        self, question: str, images: List[str]
    ) -> str:
        # model = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=500)
        model = ChatAnthropic(model="claude-3-opus-20240229")
        system_message = SystemMessage(
            content="""You are a Weights & Biases support expert.
            Your goal is to describe the attached screenshots in the context of the user query. You are provided with a support ticket and screenshots related to the issue.
            Provide a detailed description of the image in the context of the query so that the ticket can be answered correctly while incorporating the image info."""
        )
        prompt = [
            {
                "type": "text",
                "text": question,
            }
        ]
        for img in images:
            prompt += [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img}"},
                }
            ]
        message = HumanMessage(content=prompt)
        response = model.invoke([system_message, message])
        return response.content

    def __call__(
        self,
        question: str,
        chat_history: List[Tuple[str, str]] | None = None,
        images: Optional[List[str]] = None,
    ):
        if chat_history is None:
            chat_history = []

        multi_modal_response = (
            self.generate_multi_modal_initial_response(question, images)
            if images is not None
            else ""
        )

        with get_openai_callback() as query_enhancer_cb, Timer() as query_enhancer_tb:
            enhanced_query = self.query_enhancer.chain.invoke(
                {
                    "query": question,
                    "chat_history": chat_history,
                    "image_context": multi_modal_response,
                }
            )

        with Timer() as retrieval_tb:
            retrieval_results = self.retrieval.chain.invoke(enhanced_query)

        with get_openai_callback() as response_cb, Timer() as response_tb:
            response = self.response_synthesizer.chain.invoke(retrieval_results)

        output = RAGPipelineOutput(
            question=enhanced_query["standalone_query"],
            answer=response["response"],
            sources="\n".join(
                [
                    item.metadata["source"]
                    for item in retrieval_results["context"]
                ]
            ),
            source_documents=response["context_str"],
            system_prompt=response["response_prompt"],
            model=response["response_model"],
            total_tokens=query_enhancer_cb.total_tokens
            + response_cb.total_tokens,
            prompt_tokens=query_enhancer_cb.prompt_tokens
            + response_cb.prompt_tokens,
            completion_tokens=query_enhancer_cb.completion_tokens
            + response_cb.completion_tokens,
            time_taken=query_enhancer_tb.elapsed
            + retrieval_tb.elapsed
            + response_tb.elapsed,
            start_time=query_enhancer_tb.start,
            end_time=response_tb.stop,
        )

        return output
