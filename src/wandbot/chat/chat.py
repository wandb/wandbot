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

import wandb
from langchain_community.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from wandbot.chat.config import ChatConfig
from wandbot.chat.rag import load_rag_chain
from wandbot.chat.schemas import ChatRequest, ChatResponse
from wandbot.utils import Timer, get_logger

logger = get_logger(__name__)


class Chat:
    """Class for handling chat interactions.

    Attributes:
        config: An instance of ChatConfig containing configuration settings.
        run: An instance of wandb.Run for logging experiment information.
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

        self.llm = ChatOpenAI(model=self.config.chat_model_name, temperature=0)
        self.fallback_llm = ChatOpenAI(
            model="gpt-4-1106-preview", temperature=0
        )
        self.embedding_fn = OpenAIEmbeddings(
            model="text-embedding-3-small", dimensions=512
        )
        self.lang_detect_path = "data/cache/models/lid.176.bin"
        self.vector_store_path = self.config.index_artifact
        self.rag_chain = load_rag_chain(
            model=self.llm,
            fallback_model=self.fallback_llm,
            embeddings_model=self.embedding_fn,
            lang_detect_path=self.lang_detect_path,
            vector_store_path=self.vector_store_path,
            search_type="mmr",
            top_k=15,
        )

    def _get_answer(self, question, chat_history):
        result = self.rag_chain.invoke(
            {"query": question, "chat_history": chat_history}
        )

        return {
            "question": result["query"]["question"],
            "answer": result["answer"]["response"],
            "sources": "\n".join(
                [item["metadata"]["source"] for item in result["context"]]
            ),
            "source_documents": result["answer"]["context_str"],
            "system_prompt": result["answer"]["response_prompt"].to_string(),
            "model": result["answer"]["response_model"],
        }

    def __call__(self, chat_request: ChatRequest) -> ChatResponse:
        """Handles the chat request and returns the chat response.

        Args:
            chat_request: An instance of ChatRequest representing the chat request.

        Returns:
            An instance of `ChatResponse` representing the chat response.
        """
        try:
            with Timer() as timer, get_openai_callback() as oai_cb:
                result = self._get_answer(
                    chat_request.question, chat_request.chat_history
                )

            usage_stats = {
                "total_tokens": oai_cb.total_tokens,
                "prompt_tokens": oai_cb.prompt_tokens,
                "completion_tokens": oai_cb.completion_tokens,
            }
            result.update(
                dict(
                    **{
                        "question": chat_request.question,
                        "time_taken": timer.elapsed,
                        "start_time": timer.start,
                        "end_time": timer.stop,
                        "application": chat_request.application,
                    },
                    **usage_stats,
                )
            )
            self.run.log(usage_stats)
            return ChatResponse(**result)
        except Exception as e:
            with Timer() as timer:
                result = {
                    "system_prompt": "",
                    "question": chat_request.question,
                    "answer": str(e),
                    "model": "",
                    "sources": "",
                    "source_documents": "",
                    "total_tokens": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                }
            result.update(
                {
                    "time_taken": timer.elapsed,
                    "start_time": timer.start,
                    "end_time": timer.stop,
                }
            )
            usage_stats = {}
            return ChatResponse(**result)