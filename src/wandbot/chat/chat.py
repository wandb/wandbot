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
from typing import List

import weave

import wandb
from wandbot.chat.config import ChatConfig
from wandbot.chat.rag import RAGPipeline, RAGPipelineOutput
from wandbot.chat.schemas import ChatRequest, ChatResponse
from wandbot.database.schemas import QuestionAnswer
from wandbot.retriever import VectorStore
from wandbot.utils import Timer, get_logger

from openai import OpenAI

logger = get_logger(__name__)


class Chat:
    """Class for handling chat interactions.

    Attributes:
        config: An instance of ChatConfig containing configuration settings.
        run: An instance of wandb.Run for logging experiment information.
    """

    def __init__(self, vector_store: VectorStore, config: ChatConfig):
        """Initializes the Chat instance.

        Args:
            config: An instance of ChatConfig containing configuration settings.
        """
        self.vector_store = vector_store
        self.config = config
        self.run = wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            job_type="chat",
        )
        self.run._label(repo="wandbot")

        self.rag_pipeline = RAGPipeline(
            vector_store=vector_store,
            top_k=self.config.top_k,
            english_reranker_model=self.config.english_reranker_model,
            multilingual_reranker_model=self.config.multilingual_reranker_model,
            response_synthesizer_model=self.config.response_synthesizer_model,
            response_synthesizer_temperature=self.config.response_synthesizer_temperature,
            response_synthesizer_fallback_model=self.config.response_synthesizer_fallback_model,
            response_synthesizer_fallback_temperature=self.config.response_synthesizer_fallback_temperature,
        )

    def _get_answer(
        self, question: str, chat_history: List[QuestionAnswer]
    ) -> RAGPipelineOutput:
        history = []
        for item in chat_history:
            history.append(("user", item.question))
            history.append(("assistant", item.answer))

        result = self.rag_pipeline(question, history)

        return result
    
    @weave.op()
    def _translate_ja_to_en(self, text: str) -> str:
        """
        Translates Japanese text to English using OpenAI's GPT-4.

        Args:
            text: The Japanese text to be translated.

        Returns:
            The translated text in English.
        """
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a professional translator. \n\n\
                    Translate the user's question about Weights & Biases into English according to the specified rules. \n\
                    Rule of translation. \n\
                    - Maintain the original nuance\n\
                    - Keep code unchanged.\n\
                    - Only return the English translation without any additional explanation"
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0,
            max_tokens=1000,
            top_p=1
        )
        return response.choices[0].message.content
    
    @weave.op()
    def _translate_en_to_ja(self, text: str) -> str:
        """
        Translates English text to Japanese using OpenAI's GPT-4.

        Args:
            text: The English text to be translated.

        Returns:
            The translated text in Japanese.
        """
        client = OpenAI()
        response = client.chat.completions.create(
                        model="gpt-4o-2024-08-06",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a professional translator. \n\n\
                    Translate the user's text into Japanese according to the specified rules. \n\
                    Rule of translation. \n\
                    - Maintain the original nuance\n\
                    - Use 'run' in English where appropriate, as it's a term used in Wandb.\n\
                    - Translate the terms 'reference artifacts' and 'lineage' into Katakana. \n\
                    - Include specific terms in English or Katakana where appropriate\n\
                    - Keep code unchanged.\n\
                    - Only return the Japanese translation without any additional explanation"
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0,
            max_tokens=1000,
            top_p=1
        )
        return response.choices[0].message.content

    @weave.op()
    def _translate_kr_to_en(self, text: str) -> str:
        """
        Translates Korean text to English using OpenAI's GPT-4.

        Args:
            text: The Korean text to be translated.

        Returns:
            The translated text in English.
        """
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a professional translator. \n\n\
                    Translate the user's question about Weights & Biases into English according to the specified rules. \n\
                    Rule of translation. \n\
                    - Maintain the original nuance\n\
                    - Keep code unchanged.\n\
                    - Only return the English translation without any additional explanation"
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0,
            max_tokens=1000,
            top_p=1
        )
        return response.choices[0].message.content
    
    @weave.op()
    def _translate_en_to_kr(self, text: str) -> str:
        """
        Translates English text to Korean using OpenAI's GPT-4.

        Args:
            text: The English text to be translated.

        Returns:
            The translated text in Korean.
        """
        client = OpenAI()
        response = client.chat.completions.create(
                        model="gpt-4o-2024-08-06",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a professional translator. \n\n\
                    Translate the user's text into Korean according to the specified rules. \n\
                    Rule of translation. \n\
                    - Maintain the original nuance\n\
                    - Use 'run' in English where appropriate, as it's a term used in Wandb.\n\
                    - Leave the terms 'reference artifacts' and 'lineage' as English. \n\
                    - Keep code unchanged.\n\
                    - Only return the Korean translation without any additional explanation"
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0,
            max_tokens=1000,
            top_p=1
        )
        return response.choices[0].message.content
    

    @weave.op()
    def __call__(self, chat_request: ChatRequest) -> ChatResponse:
        """Handles the chat request and returns the chat response.

        Args:
            chat_request: An instance of ChatRequest representing the chat request.

        Returns:
            An instance of `ChatResponse` representing the chat response.
        """
        original_language = chat_request.language
        translation = True

        try:
            if translation:
                if original_language == "ja":
                    translated_question = self._translate_ja_to_en(chat_request.question)
                elif  original_language == "kr":
                    translated_question = self._translate_kr_to_en(chat_request.question)
                chat_request.language = "en"
                chat_request = ChatRequest(
                    question=translated_question,
                    chat_history=chat_request.chat_history,
                    application=chat_request.application,
                    language="en"
                )
                
            result = self._get_answer(
                chat_request.question, chat_request.chat_history or []
            )

            result_dict = result.model_dump()

            if translation:
                if original_language == "ja":
                    result_dict["answer"] = self._translate_en_to_ja(result_dict["answer"])
                elif original_language == "kr":
                    result_dict["answer"] = self._translate_en_to_kr(result_dict["answer"])

            usage_stats = {
                "total_tokens": result.total_tokens,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "web_search_success": result.api_call_statuses[
                    "web_search_success"
                ],
            }
            result_dict.update({"application": chat_request.application})
            self.run.log(usage_stats)
            
            return ChatResponse(**result_dict)
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

            return ChatResponse(**result)