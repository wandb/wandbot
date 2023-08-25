import json
import pathlib
from typing import List, Optional, Tuple

import langchain
import pandas as pd
import tiktoken
import wandb
from langchain import FAISS, LLMChain, OpenAI
from langchain.cache import SQLiteCache
from langchain.callbacks import get_openai_callback
from langchain.chains.conversational_retrieval.base import (
    BaseConversationalRetrievalChain,
)
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_transformers import (
    EmbeddingsClusteringFilter,
    EmbeddingsRedundantFilter,
    LongContextReorder,
)
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import (
    ContextualCompressionRetriever,
    MergerRetriever,
    MultiQueryRetriever,
    TFIDFRetriever,
)
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from wandbot.chat.config import ChatConfig
from wandbot.chat.prompts import load_chat_prompt, load_history_prompt
from wandbot.chat.schemas import ChatRepsonse, ChatRequest
from wandbot.chat.utils import ConversationalRetrievalQASourcesChain, get_chat_history
from wandbot.utils import Timer, get_logger

logger = get_logger(__name__)


class Chat:
    def __init__(self, config: ChatConfig):
        self.config = config
        self.config.llm_cache_path.parent.mkdir(parents=True, exist_ok=True)
        langchain.llm_cache = SQLiteCache(database_path=str(self.config.llm_cache_path))
        self.run = wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            job_type="chat",
        )
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        self.chat_prompt = load_chat_prompt(self.config.chat_prompt)
        self.history_prompt = load_history_prompt(self.config.history_prompt)

        self._retriever = self._load_retriever()
        self._chain = self._load_chain(
            self.config.chat_model_name, self.config.max_retries
        )
        self._fallback_chain = self._load_chain(
            self.config.fallback_model_name, self.config.max_fallback_retries
        )

    def _load_retriever(
        self,
    ):
        artifact = wandb.run.use_artifact(
            self.config.retriever_artifact, type="vectorstore"
        )
        artifact_dir = artifact.download()

        embedding_fn = OpenAIEmbeddings()
        retrievers = []
        for docstore_dir in list(pathlib.Path(artifact_dir).glob("*")):
            faiss_dir = docstore_dir / "faiss"
            tftdf_dir = docstore_dir / "tfidf"
            faiss_store = FAISS.load_local(str(faiss_dir), embedding_fn)
            faiss_retriever = faiss_store.as_retriever()
            tfidf_retriever = TFIDFRetriever.load_local(str(tftdf_dir))
            retrievers.extend([faiss_retriever, tfidf_retriever])
        merger_retriever = MergerRetriever(retrievers=retrievers)
        query_llm = ChatOpenAI(model=self.config.fallback_model_name, temperature=0.5)
        retriever_from_llm = MultiQueryRetriever.from_llm(
            retriever=merger_retriever, llm=query_llm
        )
        embedding_filter = EmbeddingsRedundantFilter(embeddings=embedding_fn)

        filter_ordered_cluster = EmbeddingsClusteringFilter(
            embeddings=embedding_fn,
            num_clusters=5,
            num_closest=2,
        )
        pipeline = DocumentCompressorPipeline(
            transformers=[
                embedding_filter,
                filter_ordered_cluster,
                embedding_filter,
                LongContextReorder(),
            ]
        )
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline, base_retriever=retriever_from_llm
        )
        return compression_retriever

    @property
    def retriever(self):
        if self._retriever is None:
            self._retriever = self._load_retriever()
        return self._retriever

    def _load_chain(
        self, model_name: str = None, max_retries: int = 1
    ) -> BaseConversationalRetrievalChain:
        map_llm = OpenAI(
            batch_size=10,
            temperature=0.0,
            max_retries=self.config.max_fallback_retries,
        )
        reduce_llm = ChatOpenAI(
            model_name=self.config.fallback_model_name,
            temperature=self.config.chat_temperature,
            max_retries=max_retries,
        )
        question_generator = LLMChain(
            llm=ChatOpenAI(
                model_name=self.config.fallback_model_name,
                temperature=self.config.chat_temperature,
                max_retries=self.config.max_fallback_retries,
            ),
            prompt=self.history_prompt,
            verbose=self.config.verbose,
        )
        if self.config.chain_type == "map_reduce":
            doc_chain = load_qa_with_sources_chain(
                map_llm,
                chain_type=self.config.chain_type,
                combine_prompt=self.chat_prompt,
                verbose=self.config.verbose,
                reduce_llm=reduce_llm,
            )
        else:
            doc_chain = load_qa_with_sources_chain(
                reduce_llm,
                chain_type=self.config.chain_type,
                prompt=self.chat_prompt,
                verbose=self.config.verbose,
            )
        chain = ConversationalRetrievalQASourcesChain(
            retriever=self.retriever,
            question_generator=question_generator,
            combine_docs_chain=doc_chain,
            return_source_documents=True,
        )

        return chain

    @property
    def chain(self):
        if self._chain is None:
            self._chain = self._load_chain(
                model_name=self.config.chat_model_name,
                max_retries=self.config.max_retries,
            )
        return self._chain

    @property
    def fallback_chain(self):
        if self._fallback_chain is None:
            self._fallback_chain = self._load_chain(
                model_name=self.config.fallback_model_name,
                max_retries=self.config.max_fallback_retries,
            )
        return self._fallback_chain

    def validate_and_format_question(self, question: str) -> str:
        question = " ".join(question.strip().split())

        if len(self.tokenizer.encode(question)) > 1024:
            raise ValueError(
                f"Question is too long. Please rephrase your question to be shorter than {1024 * 3 // 4} words."
            )
        return question

    def format_response(self, result):
        response = {}
        if result["source_documents"]:
            source_documents = [
                {
                    "source": doc.metadata["source"],
                    "text": doc.page_content,
                }
                for doc in result["source_documents"]
            ]
        else:
            source_documents = []
        response["answer"] = result["answer"]
        response["model"] = result["model"]

        if len(source_documents) and self.config.include_sources:
            response["source_documents"] = json.dumps(source_documents)
            response["sources"] = result["sources"]
        else:
            response["source_documents"] = ""
            response["sources"] = ""

        return response

    def get_answer(
        self, query: str, chat_history: Optional[List[Tuple[str, str]]] = None
    ):
        if chat_history is None:
            chat_history = []
        try:
            result = self.chain(
                {
                    "question": query,
                    "chat_history": chat_history,
                },
                return_only_outputs=True,
            )
            result["model"] = self.config.chat_model_name
        except Exception as e:
            logger.warning(f"{self.config.chat_model_name} failed with {e}")
            logger.warning(f"Falling back to {self.config.fallback_model_name} model")
            try:
                result = self.fallback_chain(
                    {
                        "question": query,
                        "chat_history": chat_history,
                    },
                    return_only_outputs=True,
                )
                result["model"] = self.config.fallback_model_name
            except Exception as e:
                logger.warning(f"{self.config.fallback_model_name} failed with {e}")
                result = {
                    "answer": "\uE058"
                    + " Sorry, there seems to be an issue with our LLM service. Please try again in some time.",
                    "sources": "",
                    "source_documents": None,
                    "model": "None",
                }
        return self.format_response(result)

    def __call__(self, chat_request: ChatRequest) -> ChatRepsonse:
        with Timer() as timer:
            try:
                query = self.validate_and_format_question(chat_request.question)
            except ValueError as e:
                result = {
                    "answer": str(e),
                    "sources": "",
                }
            else:
                with get_openai_callback() as callback:
                    result = self.get_answer(
                        query, chat_history=get_chat_history(chat_request.chat_history)
                    )
                    usage_stats = {
                        "total_tokens": callback.total_tokens,
                        "prompt_tokens": callback.prompt_tokens,
                        "completion_tokens": callback.completion_tokens,
                        "successful_requests": callback.successful_requests,
                        "total_cost": callback.total_cost,
                    }

        result.update(
            dict(
                **{
                    "question": chat_request.question,
                    "time_taken": timer.elapsed,
                    "start_time": timer.start,
                    "end_time": timer.stop,
                },
                **usage_stats,
            )
        )
        wandb.run.log(
            {
                "chat": wandb.Table(
                    dataframe=pd.DataFrame({k: [v] for k, v in result.items()})
                )
            }
        )
        wandb.run.log(usage_stats)
        return ChatRepsonse(**result)


def main():
    config = ChatConfig()
    chat = Chat(config=config)
    chat_history = []
    while True:
        question = input("You: ")
        if question.lower() == "quit":
            break
        else:
            response = chat(ChatRequest(question=question, chat_history=chat_history))
            chat_history.append((question, response.answer))
            print(f"WandBot: {response.answer}")
            print(f"Time taken: {response.time_taken}")
