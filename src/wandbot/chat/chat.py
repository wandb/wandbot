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
from typing import Any, Dict, List, Optional, Tuple

import wandb
from llama_index import ServiceContext
from llama_index.callbacks import (
    CallbackManager,
    TokenCountingHandler,
    WandbCallbackHandler,
    trace_method,
)
from llama_index.chat_engine import ContextChatEngine
from llama_index.chat_engine.types import AgentChatResponse
from llama_index.indices.postprocessor import CohereRerank
from llama_index.llms import ChatMessage, MessageRole
from llama_index.llms.generic_utils import messages_to_history_str
from llama_index.schema import MetadataMode, NodeWithScore, QueryBundle
from llama_index.tools import ToolOutput
from wandbot.chat.config import ChatConfig
from wandbot.chat.prompts import load_chat_prompt, partial_format
from wandbot.chat.query_enhancer import CompleteQuery, QueryHandler
from wandbot.chat.retriever import (
    LanguageFilterPostprocessor,
    MetadataPostprocessor,
    Retriever,
)
from wandbot.chat.schemas import ChatRequest, ChatResponse
from wandbot.utils import Timer, get_logger, load_service_context
from weave.monitoring import StreamTable

logger = get_logger(__name__)


def rebuild_full_prompt(
    message_templates: List[ChatMessage], result: Dict[str, Any]
) -> str:
    system_template = messages_to_history_str(message_templates[:-1])

    query_str = result["question"]

    context = json.loads(result["source_documents"])

    context_str = ""
    for item in context:
        context_str += "source: " + item["source"] + "\n\n"
        context_str += item["text"] + "\n\n"
        context_str += "---\n\n"

    query_content = partial_format(
        message_templates[-1].content,
        query_str=query_str,
        context_str=context_str,
    )
    system_template += (
        f"\n# {message_templates[-1].role}:\n\t{query_content}\n\n"
    )

    return system_template


class WandbContextChatEngine(ContextChatEngine):
    def _generate_context(
        self, message: str, **kwargs
    ) -> Tuple[str, List[NodeWithScore]]:
        """Generate context information from a message."""

        keywords = kwargs.get("keywords", [])
        sub_queries = kwargs.get("sub_queries", [])

        query_nodes = self._retriever.retrieve(message)
        keywords_nodes = []
        sub_query_nodes = []

        if keywords:
            keywords_nodes = self._retriever.retrieve(" ".join(keywords))

        if sub_queries:
            for sub_query in sub_queries:
                sub_query_nodes += self._retriever.retrieve(sub_query)

        nodes = query_nodes + keywords_nodes + sub_query_nodes

        for postprocessor in self._node_postprocessors:
            nodes = postprocessor.postprocess_nodes(
                nodes, query_bundle=QueryBundle(message)
            )

        context_str = "\n\n---\n\n".join(
            [
                n.node.get_content(metadata_mode=MetadataMode.LLM).strip()
                for n in nodes
            ]
        )

        return context_str.strip(), nodes

    def _get_prefix_messages_with_context(
        self, context_str: str
    ) -> List[ChatMessage]:
        """Get the prefix messages with context."""
        prefix_messages = self._prefix_messages

        context_str_w_sys_prompt = partial_format(
            prefix_messages[-1].content, context_str=context_str
        )
        return [
            *prefix_messages[:-1],
            ChatMessage(
                content=context_str_w_sys_prompt,
                role=MessageRole.USER,
                metadata={},
            ),
        ]

    @trace_method("chat")
    def chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        **kwargs,
    ) -> AgentChatResponse:
        context_str_template, nodes = self._generate_context(
            message,
            keywords=kwargs.get("keywords", []),
            sub_queries=kwargs.get("sub_queries", []),
        )
        prefix_messages = self._get_prefix_messages_with_context(
            context_str_template
        )

        prefix_messages[-1] = ChatMessage(
            content=partial_format(
                prefix_messages[-1].content, query_str=message
            ),
            role="user",
        )

        self._memory.put(prefix_messages[-1])
        all_messages = prefix_messages
        chat_response = self._llm.chat(all_messages)
        ai_message = chat_response.message
        self._memory.put(ai_message)

        return AgentChatResponse(
            response=str(chat_response.message.content),
            sources=[
                ToolOutput(
                    tool_name="retriever",
                    content=str(prefix_messages[0]),
                    raw_input={"message": message},
                    raw_output=prefix_messages[0],
                )
            ],
            source_nodes=nodes,
        )


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
        self.chat_table = StreamTable(
            table_name="chat_logs",
            project_name=self.config.wandb_project,
            entity_name=self.config.wandb_entity,
            # f"{self.config.wandb_entity}/{self.config.wandb_project}/chat_logs"
        )

        self.wandb_callback = WandbCallbackHandler()
        self.token_counter = TokenCountingHandler()
        self.callback_manager = CallbackManager(
            [self.wandb_callback, self.token_counter]
        )
        self.default_service_context = load_service_context(
            llm=self.config.chat_model_name,
            temperature=self.config.chat_temperature,
            max_retries=self.config.max_retries,
            embeddings_cache=str(self.config.embeddings_cache),
            callback_manager=self.callback_manager,
        )
        self.fallback_service_context = load_service_context(
            llm=self.config.fallback_model_name,
            temperature=self.config.chat_temperature,
            max_retries=self.config.max_fallback_retries,
            embeddings_cache=str(self.config.embeddings_cache),
            callback_manager=self.callback_manager,
        )

        self.qa_prompt = load_chat_prompt(f_name=self.config.chat_prompt)
        self.query_handler = QueryHandler()
        self.retriever = Retriever(
            run=self.run,
            service_context=self.fallback_service_context,
            callback_manager=self.callback_manager,
        )

    def _load_chat_engine(
        self,
        service_context: ServiceContext,
        query_intent: str = "\n",
        language: str = "en",
        initial_k: int = 15,
        top_k: int = 5,
    ) -> WandbContextChatEngine:
        """Loads the chat engine with the given model name and maximum retries.

        Args:
            service_context: An instance of ServiceContext.
            query_intent: A string representing the query intent.
            language: A string representing the language.
            initial_k: An integer representing the initial number of documents to retrieve.
            top_k: An integer representing the number of documents to retrieve after reranking.

        Returns:
            An instance of ChatEngine.
        """

        query_engine = self.retriever.load_query_engine(
            similarity_top_k=initial_k,
            language=language,
            top_k=top_k,
        )

        self.qa_prompt = load_chat_prompt(
            f_name=self.config.chat_prompt,
            language_code=language,
            query_intent=query_intent,
        )
        chat_engine_kwargs = dict(
            retriever=query_engine.retriever,
            storage_context=self.retriever.storage_context,
            service_context=service_context,
            similarity_top_k=initial_k,
            response_mode="compact",
            node_postprocessors=[
                LanguageFilterPostprocessor(languages=[language, "python"]),
                CohereRerank(top_n=top_k, model="rerank-english-v2.0")
                if language == "en"
                else CohereRerank(
                    top_n=top_k, model="rerank-multilingual-v2.0"
                ),
                MetadataPostprocessor(),
            ],
            prefix_messages=self.qa_prompt.message_templates,
        )

        chat_engine = WandbContextChatEngine.from_defaults(**chat_engine_kwargs)

        return chat_engine

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

    def get_response(
        self,
        service_context: ServiceContext,
        query: str,
        language: str,
        chat_history: List[ChatMessage],
        query_intent: str,
        keywords: List[str] | None = None,
        sub_queries: List[str] | None = None,
    ) -> Dict[str, Any]:
        chat_engine = self._load_chat_engine(
            service_context=service_context,
            language=language,
            query_intent=query_intent,
        )
        response = chat_engine.chat(
            message=query,
            chat_history=chat_history,
            keywords=keywords,
            sub_queries=sub_queries,
        )
        result = {
            "answer": response.response,
            "source_documents": response.source_nodes,
            "model": self.config.chat_model_name,
        }
        return result

    def get_answer(
        self,
        resolved_query: CompleteQuery,
        **kwargs,
    ) -> Dict[str, Any]:
        """Gets the answer for the given query and chat history.

        Args:
            resolved_query: An instance of ResolvedQuery representing the resolved query.

        Returns:
            A dictionary representing the answer.

        """
        try:
            result = self.get_response(
                service_context=self.default_service_context,
                query=resolved_query.condensed_query,
                language=resolved_query.language,
                chat_history=resolved_query.chat_history,
                query_intent=resolved_query.intent_hints,
            )
        except Exception as e:
            logger.warning(f"{self.config.chat_model_name} failed with {e}")
            logger.warning(
                f"Falling back to {self.config.fallback_model_name} model"
            )
            try:
                result = self.get_response(
                    service_context=self.fallback_service_context,
                    query=resolved_query.cleaned_query,
                    language=resolved_query.language,
                    chat_history=resolved_query.chat_history,
                    query_intent=resolved_query.intent_hints,
                )

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
            An instance of `ChatResponse` representing the chat response.
        """
        with Timer() as timer:
            result = {}
            try:
                resolved_query = self.query_handler(chat_request)
                result = self.get_answer(resolved_query)
                usage_stats = {
                    "total_tokens": self.token_counter.total_llm_token_count,
                    "prompt_tokens": self.token_counter.prompt_llm_token_count,
                    "completion_tokens": self.token_counter.completion_llm_token_count,
                }
                self.token_counter.reset_counts()
            except ValueError as e:
                result = {
                    "answer": str(e),
                    "sources": "",
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

        system_template = rebuild_full_prompt(
            self.qa_prompt.message_templates, result
        )
        result["system_prompt"] = system_template
        self.chat_table.log(result)
        return ChatResponse(**result)
