import json
from typing import List, Optional, Any, Dict

import tiktoken
import wandb
from llama_index import StorageContext, load_index_from_storage
from llama_index.callbacks import (
    WandbCallbackHandler,
    TokenCountingHandler,
    CallbackManager,
)
from llama_index.chat_engine.types import ChatMode
from llama_index.indices.postprocessor import CohereRerank
from llama_index.llms import ChatMessage, MessageRole
from llama_index.vector_stores import FaissVectorStore

from wandbot.chat.config import ChatConfig
from wandbot.chat.prompts import load_chat_prompt
from wandbot.chat.schemas import ChatRepsonse, ChatRequest
from wandbot.database.schemas import QuestionAnswer
from wandbot.utils import Timer, get_logger, load_service_context

logger = get_logger(__name__)


def get_chat_history(
    chat_history: List[QuestionAnswer] | None,
) -> Optional[List[ChatMessage]]:
    if not chat_history:
        return None
    else:
        messages = [
            [
                ChatMessage(role=MessageRole.USER, content=question_answer.question),
                ChatMessage(role=MessageRole.ASSISTANT, content=question_answer.answer),
            ]
            for question_answer in chat_history
        ]
        return [item for sublist in messages for item in sublist]


class Chat:
    def __init__(self, config: ChatConfig):
        self.config = config
        self.run = wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            job_type="chat",
        )
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        self.storage_context = self.load_storage_context_from_artifact(
            artifact_url=self.config.index_artifact
        )
        self.index = load_index_from_storage(self.storage_context)

        self.wandb_callback = WandbCallbackHandler()
        self.token_counter = TokenCountingHandler(tokenizer=self.tokenizer.encode)
        self.callback_manager = CallbackManager(
            [self.wandb_callback, self.token_counter]
        )

        self.qa_prompt = load_chat_prompt(self.config.chat_prompt)
        self.chat_engine = self._load_chat_engine(
            self.config.chat_model_name,
            max_retries=self.config.max_retries,
        )
        self.fallback_chat_engine = self._load_chat_engine(
            self.config.fallback_model_name,
            max_retries=self.config.max_fallback_retries,
        )

    def load_storage_context_from_artifact(self, artifact_url: str):
        artifact = self.run.use_artifact(artifact_url)
        artifact_dir = artifact.download()
        storage_context = StorageContext.from_defaults(
            vector_store=FaissVectorStore.from_persist_dir(artifact_dir),
            persist_dir=artifact_dir,
        )
        return storage_context

    def _load_chat_engine(self, model_name, max_retries):
        service_context = load_service_context(
            model_name,
            temperature=self.config.chat_temperature,
            max_retries=max_retries,
            embeddings_cache=self.config.embeddings_cache,
            callback_manager=self.callback_manager,
        )
        chat_engine = self.index.as_chat_engine(
            chat_mode=ChatMode.CONDENSE_QUESTION,
            similarity_top_k=20,
            response_mode="compact",
            service_context=service_context,
            text_qa_template=self.qa_prompt,
            node_postprocessors=[CohereRerank(top_n=10, model="rerank-english-v2.0")],
            storage_context=self.storage_context,
        )
        return chat_engine

    def validate_and_format_question(self, question: str) -> str:
        question = " ".join(question.strip().split())

        if len(self.tokenizer.encode(question)) > 1024:
            raise ValueError(
                f"Question is too long. Please rephrase your question to be shorter than {1024 * 3 // 4} words."
            )
        return question

    def format_response(self, result: Dict[str, Any]):
        response = {}
        if result.get("source_documents", None):
            source_documents = [
                {
                    "source": doc.metadata["source"],
                    "text": doc.text,
                }
                for doc in result["source_documents"]
            ]
        else:
            source_documents = []
        response["answer"] = result["answer"]
        response["model"] = result["model"]

        if len(source_documents) and self.config.include_sources:
            response["source_documents"] = json.dumps(source_documents)
            response["sources"] = ",".join([doc["source"] for doc in source_documents])
        else:
            response["source_documents"] = ""
            response["sources"] = ""

        return response

    def get_answer(self, query: str, chat_history: Optional[List[ChatMessage]] = None):
        try:
            response = self.chat_engine.chat(message=query, chat_history=chat_history)
            result = {
                "answer": response.response,
                "source_documents": response.source_nodes,
                "model": self.config.chat_model_name,
            }
        except Exception as e:
            logger.warning(f"{self.config.chat_model_name} failed with {e}")
            logger.warning(f"Falling back to {self.config.fallback_model_name} model")
            try:
                response = self.fallback_chat_engine.chat(
                    message=query, chat_history=chat_history
                )
                result = {
                    "answer": response.response,
                    "source_documents": response.source_nodes,
                    "model": self.config.fallback_model_name,
                }

            except Exception as e:
                logger.warning(f"{self.config.fallback_model_name} failed with {e}")
                result = {
                    "answer": "\uE058"
                    + " Sorry, there seems to be an issue with our LLM service. Please try again in some time.",
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
                result = self.get_answer(
                    query, chat_history=get_chat_history(chat_request.chat_history)
                )
                usage_stats = {
                    "total_tokens": self.token_counter.total_llm_token_count,
                    "prompt_tokens": self.token_counter.prompt_llm_token_count,
                    "completion_tokens": self.token_counter.completion_llm_token_count,
                }
                self.token_counter.reset_counts()

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
        self.run.log(usage_stats)
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
            chat_history.append(
                QuestionAnswer(question=question, answer=response.answer)
            )
            print(f"WandBot: {response.answer}")
            print(f"Time taken: {response.time_taken}")


if __name__ == "__main__":
    main()
