"""Handles chat interactions for WandBot.

This module contains the Chat class which is responsible for handling chat interactions. 
It includes methods for initializing the chat, loading the storage context from an artifact, 
loading the chat engine, validating and formatting questions, formatting responses, and getting answers. 
It also contains a function for generating a list of chat messages from a given chat history.

Typical usage example:

  config = ChatConfig()
  chat = Chat(config=config)
  chat_history = []
  while True:
      question = input("You: ")
      if question.lower() == "quit":
          break
      else:
          response = chat(
              ChatRequest(question=question, chat_history=chat_history)
          )
          chat_history.append(
              QuestionAnswer(question=question, answer=response.answer)
          )
          print(f"WandBot: {response.answer}")
          print(f"Time taken: {response.time_taken}")
"""

import json
from typing import Any, Dict, List, Optional

import tiktoken
from llama_index import StorageContext, load_index_from_storage
from llama_index.callbacks import (
    CallbackManager,
    TokenCountingHandler,
    WandbCallbackHandler,
)
from llama_index.chat_engine.types import BaseChatEngine, ChatMode
from llama_index.indices.postprocessor import CohereRerank
from llama_index.llms import ChatMessage, MessageRole
from llama_index.vector_stores import FaissVectorStore
from llama_index.vector_stores.simple import DEFAULT_VECTOR_STORE, NAMESPACE_SEP
from llama_index.vector_stores.types import DEFAULT_PERSIST_FNAME
from weave.monitoring import StreamTable

import wandb
from wandbot.chat.config import ChatConfig
from wandbot.chat.prompts import load_chat_prompt
from wandbot.chat.schemas import ChatRequest, ChatResponse
from wandbot.database.schemas import QuestionAnswer
from wandbot.utils import (
    LanguageFilterPostprocessor,
    Timer,
    get_logger,
    load_service_context,
)

logger = get_logger(__name__)


def get_chat_history(
    chat_history: List[QuestionAnswer] | None,
) -> Optional[List[ChatMessage]]:
    """Generates a list of chat messages from a given chat history.

    This function takes a list of QuestionAnswer objects and transforms them into a list of ChatMessage objects. Each
    QuestionAnswer object is split into two ChatMessage objects: one for the user's question and one for the
    assistant's answer. If the chat history is empty or None, the function returns None.

    Args: chat_history: A list of QuestionAnswer objects representing the history of a chat. Each QuestionAnswer
    object contains a question from the user and an answer from the assistant.

    Returns: A list of ChatMessage objects representing the chat history. Each ChatMessage object has a role (either
    'USER' or 'ASSISTANT') and content (the question or answer text). If the chat history is empty or None,
    the function returns None.
    """
    if not chat_history:
        return None
    else:
        messages = [
            [
                ChatMessage(
                    role=MessageRole.USER, content=question_answer.question
                ),
                ChatMessage(
                    role=MessageRole.ASSISTANT, content=question_answer.answer
                ),
            ]
            for question_answer in chat_history
        ]
        return [item for sublist in messages for item in sublist]


class Chat:
    """Class for handling chat interactions.

    Attributes:
        config: An instance of ChatConfig containing configuration settings.
        run: An instance of wandb.Run for logging experiment information.
        tokenizer: An instance of tiktoken.Tokenizer for encoding text.
        storage_context: An instance of StorageContext for managing storage.
        index: An instance of Index for storing and retrieving vectors.
        wandb_callback: An instance of WandbCallbackHandler for handling Wandb callbacks.
        token_counter: An instance of TokenCountingHandler for counting tokens.
        callback_manager: An instance of CallbackManager for managing callbacks.
        qa_prompt: A string representing the chat prompt.
    """

    def __init__(self, config: ChatConfig):
        """Initializes the Chat instance.

        Args:
            config: An instance of ChatConfig containing configuration settings.
        """
        self.config = config
        self.run = wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            job_type="chat",
        )
        self.run._label(repo="wandbot")
        self.chat_table = StreamTable(
            f"{self.config.wandb_entity}/{self.config.wandb_project}/chat_logs"
        )
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        self.storage_context = self.load_storage_context_from_artifact(
            artifact_url=self.config.index_artifact
        )
        self.index = load_index_from_storage(self.storage_context)

        self.wandb_callback = WandbCallbackHandler()
        self.token_counter = TokenCountingHandler(
            tokenizer=self.tokenizer.encode
        )
        self.callback_manager = CallbackManager(
            [self.wandb_callback, self.token_counter]
        )

        self.qa_prompt = load_chat_prompt(self.config.chat_prompt)

    def load_storage_context_from_artifact(
        self, artifact_url: str
    ) -> StorageContext:
        """Loads the storage context from the given artifact URL.

        Args:
            artifact_url: A string representing the URL of the artifact.

        Returns:
            An instance of StorageContext.
        """
        artifact = self.run.use_artifact(artifact_url)
        artifact_dir = artifact.download()
        index_path = f"{artifact_dir}/{DEFAULT_VECTOR_STORE}{NAMESPACE_SEP}{DEFAULT_PERSIST_FNAME}"
        storage_context = StorageContext.from_defaults(
            vector_store=FaissVectorStore.from_persist_path(index_path),
            persist_dir=artifact_dir,
        )
        return storage_context

    def _load_chat_engine(
        self,
        model_name: str,
        max_retries: int,
        has_chat_history: bool = False,
        language: str = "en",
    ) -> BaseChatEngine:
        """Loads the chat engine with the given model name and maximum retries.

        Args:
            model_name: A string representing the name of the model.
            max_retries: An integer representing the maximum number of retries.

        Returns:
            An instance of ChatEngine.
        """
        service_context = load_service_context(
            model_name,
            temperature=self.config.chat_temperature,
            max_retries=max_retries,
            embeddings_cache=str(self.config.embeddings_cache),
            callback_manager=self.callback_manager,
        )

        chat_engine = self.index.as_chat_engine(
            chat_mode=ChatMode.CONDENSE_QUESTION
            if has_chat_history
            else ChatMode.CONTEXT,
            similarity_top_k=50,
            response_mode="compact",
            service_context=service_context,
            text_qa_template=self.qa_prompt,
            node_postprocessors=[
                LanguageFilterPostprocessor(languages=[language, "python"]),
                CohereRerank(top_n=5, model="rerank-multilingual-v2.0")
                if language == "ja"
                else CohereRerank(top_n=5, model="rerank-english-v2.0"),
            ],
            storage_context=self.storage_context,
        )
        return chat_engine

    def validate_and_format_question(self, question: str) -> str:
        """Validates and formats the given question.

        Args:
            question: A string representing the question to validate and format.

        Returns:
            A string representing the validated and formatted question.

        Raises:
            ValueError: If the question is too long.
        """
        question = " ".join(question.strip().split())

        if len(self.tokenizer.encode(question)) > 1024:
            raise ValueError(
                f"Question is too long. Please rephrase your question to be shorter than {1024 * 3 // 4} words."
            )
        return question

    def format_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Formats the response dictionary.

        Args:
            result: A dictionary representing the response.

        Returns:
            A formatted response dictionary.
        """
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
            response["sources"] = ",".join(
                [doc["source"] for doc in source_documents]
            )
        else:
            response["source_documents"] = ""
            response["sources"] = ""

        return response

    def get_answer(
        self,
        query: str,
        chat_history: Optional[List[ChatMessage]] = None,
        language: str = "en",
    ) -> Dict[str, Any]:
        """Gets the answer for the given query and chat history.

        Args:
            query: A string representing the query.
            chat_history: A list of ChatMessage representing the chat history.
            language: A string representing the language of the query.

        Returns:
            A formatted response dictionary.
        """
        try:
            chat_engine = self._load_chat_engine(
                self.config.chat_model_name,
                max_retries=self.config.max_retries,
                language=language,
                has_chat_history=bool(chat_history),
            )
            response = chat_engine.chat(
                message=query, chat_history=chat_history
            )
            result = {
                "answer": response.response,
                "source_documents": response.source_nodes,
                "model": self.config.chat_model_name,
            }
        except Exception as e:
            logger.warning(f"{self.config.chat_model_name} failed with {e}")
            logger.warning(
                f"Falling back to {self.config.fallback_model_name} model"
            )
            try:
                fallback_chat_engine = self._load_chat_engine(
                    self.config.fallback_model_name,
                    max_retries=self.config.max_fallback_retries,
                    language=language,
                    has_chat_history=bool(chat_history),
                )
                response = fallback_chat_engine.chat(
                    message=query,
                    chat_history=chat_history,
                )
                result = {
                    "answer": response.response,
                    "source_documents": response.source_nodes,
                    "model": self.config.fallback_model_name,
                }

            except Exception as e:
                logger.error(
                    f"{self.config.fallback_model_name} failed with {e}"
                )
                result = {
                    "answer": "\uE058"
                    + " Sorry, there seems to be an issue with our LLM service. Please try again in some time.",
                    "source_documents": None,
                    "model": "None",
                }
        return self.format_response(result)

    def __call__(self, chat_request: ChatRequest) -> ChatResponse:
        """Handles the chat request and returns the chat response.

        Args:
            chat_request: An instance of ChatRequest representing the chat request.

        Returns:
            An instance of ChatResponse representing the chat response.
        """
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
                    query=query,
                    chat_history=get_chat_history(chat_request.chat_history),
                    language=chat_request.language,
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
        result["system_prompt"] = self.qa_prompt.message_templates[0].content
        self.chat_table.log(result)
        return ChatResponse(**result)


def main():
    config = ChatConfig()
    chat = Chat(config=config)
    chat_history = []
    while True:
        question = input("You: ")
        if question.lower() == "quit":
            break
        else:
            response = chat(
                ChatRequest(question=question, chat_history=chat_history)
            )
            chat_history.append(
                QuestionAnswer(question=question, answer=response.answer)
            )
            print(f"WandBot: {response.answer}")
            print(f"Time taken: {response.time_taken}")


if __name__ == "__main__":
    main()
