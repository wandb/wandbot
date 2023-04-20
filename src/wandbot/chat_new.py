import pathlib
from typing import Any, Dict, List, Optional, Tuple

import tiktoken
from langchain.chains.conversational_retrieval.base import (
    BaseConversationalRetrievalChain,
)
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseRetriever
from pydantic import BaseModel
from src.wandbot.customization.langchain import (
    ConversationalRetrievalQAWithSourcesChainWithScore,
)
from src.wandbot.ingestion.utils import Timer
from src.wandbot.prompts import load_chat_prompt
from wandbot.ingestion.datastore import VectorIndex
from wandbot.ingestion.settings import VectorIndexConfig


class ChatConfig(BaseModel):
    model_name: str = "gpt-4"
    max_retries: int = 1
    fallback_model_name: str = "gpt-3.5-turbo"
    max_fallback_retries: int = 6
    chat_temperature: float = 0.0
    chain_type: str = "stuff"
    chat_prompt: pathlib.Path = pathlib.Path("data/prompts/chat_prompt.txt")
    vector_index_config: VectorIndexConfig = VectorIndexConfig(
        wandb_project="wandb_docs_bot_dev"
    )
    vector_index_artifact: str = (
        "parambharat/wandb_docs_bot_dev/wandbot_vectorindex:latest"
    )
    wandb_project: str = "wandb_docs_bot_dev"
    wandb_entity: str = "wandb"
    respond_with_sources: bool = True
    source_score_threshold: float = 1.0
    query_tokens_threshold: int = 1024


class Chat:
    config: ChatConfig

    def __init__(self, config: Optional[ChatConfig] = None):
        if config is not None:
            self.config: ChatConfig = config
        self.vector_index: VectorIndex = VectorIndex(
            config=self.config.vector_index_config
        )
        self.chat_prompt: ChatPromptTemplate = load_chat_prompt(self.config.chat_prompt)
        self._retriever: BaseRetriever = self._load_retriever()
        self._chain: BaseConversationalRetrievalChain = self._load_chain(
            self.config.model_name, self.config.max_retries
        )
        self._fallback_chain: BaseConversationalRetrievalChain = self._load_chain(
            self.config.fallback_model_name, self.config.max_fallback_retries
        )
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _load_retriever(self) -> BaseRetriever:
        self.vector_index = self.vector_index.load_from_artifact(
            self.config.vector_index_artifact
        )
        return self.vector_index.retriever

    @property
    def retriever(self):
        if self._retriever is None:
            self._retriever = self._load_retriever()
        return self._retriever

    def _load_chain(
        self, model_name: str = None, max_retries: int = 1
    ) -> BaseConversationalRetrievalChain:
        chain = ConversationalRetrievalQAWithSourcesChainWithScore.from_llm(
            ChatOpenAI(
                model_name=model_name,
                temperature=self.config.chat_temperature,
                max_retries=max_retries,
            ),
            chain_type="stuff",
            retriever=self.retriever,
            qa_prompt=self.chat_prompt,
            return_source_documents=True,
            verbose=True,
        )
        return chain

    @property
    def chain(self):
        if self._chain is None:
            self._chain = self._load_chain(
                model_name=self.config.model_name,
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

    def get_answer(
        self, query: str, chat_history: Optional[List[Tuple[str, str]]] = None
    ):
        used_fallback = False
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
        except Exception as e:
            result = self.fallback_chain(
                {
                    "question": query,
                    "chat_history": chat_history,
                },
                return_only_outputs=True,
            )
            used_fallback = True

        return self.format_response(result, used_fallback)

    def format_response(self, result, used_fallback: bool):
        sources = list(
            {
                "- " + doc.metadata["source"]
                for doc in result["source_documents"]
                # if doc.metadata["score"] <= self.config.source_score_threshold
            }
        )

        if len(sources) and self.config.respond_with_sources:
            response = result["answer"] + "\n\n*References*:\n\n" + "\n".join(sources)
        else:
            response = result["answer"]

        if used_fallback:
            response = (
                f"**Warning: Falling back to {self.config.fallback_model_name}.** "
                f"These results are sometimes not as good as {self.config.model_name} \n\n"
                + response
            )

        return response

    def __call__(
        self, question: str, chat_history: Optional[List[Tuple[str, str]]] = None
    ) -> Dict[str, Any]:
        with Timer() as timer:
            try:
                query = self.validate_and_format_question(question)
            except ValueError as e:
                return {
                    "question": question,
                    "response": str(e),
                    "time_taken": timer.elapsed,
                }
            response = self.get_answer(query, chat_history=chat_history)
            return {
                "question": question,
                "response": response,
                "time_taken": timer.elapsed,
            }


def main():
    config = ChatConfig()
    chat = Chat(config=config)
    chat_history = []
    while True:
        question = input("You: ")
        if question.lower() == "quit":
            break
        else:
            response = chat(question, chat_history=chat_history)
            chat_history.append((question, response["response"]))
            print(f"WandBot: {response['response']}")
            print(f"Time taken: {response['time_taken']}")
            print("")


if __name__ == "__main__":
    main()
