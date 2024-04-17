from operator import itemgetter

import regex as re
from langchain_core.messages import convert_to_messages, messages_to_dict
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from wandbot.rag.query_handler.history_handler import CondenseQuestion
from wandbot.rag.query_handler.intents_enhancer import IntentsEnhancer
from wandbot.rag.query_handler.keyword_search_enhancer import KeywordsEnhancer
from wandbot.rag.query_handler.language_detection import LanguageDetector
from wandbot.rag.query_handler.vector_search_enhancer import (
    VectorSearchEnhancer,
)
from wandbot.rag.query_handler.web_search import YouWebRagSearchEnhancer

BOT_NAME_PATTERN = re.compile(r"<@U[A-Z0-9]+>|@[a-zA-Z0-9]+")


def clean_question(question: str) -> str:
    cleaned_query = BOT_NAME_PATTERN.sub("", question).strip()
    return cleaned_query


class QueryEnhancer:
    def __init__(
        self,
        model: str = "gpt-4-1106-preview",
        fallback_model="gpt-4-1106-preview",
    ):
        self.question_condenser = CondenseQuestion(
            model=model, fallback_model=fallback_model
        )
        self.intents_enhancer = IntentsEnhancer(
            model=model, fallback_model=fallback_model
        )
        self.language_detector = LanguageDetector()
        self.keywords_enhancer = KeywordsEnhancer(
            model=model, fallback_model=fallback_model
        )
        self.vector_search_enhancer = VectorSearchEnhancer(
            model=model, fallback_model=fallback_model
        )
        self.web_search_enhancer = YouWebRagSearchEnhancer()

        self._chain = None

    @property
    def chain(self) -> Runnable:
        if self._chain is None:
            self._chain = self._load_chain()
        return self._chain

    def _load_chain(self) -> Runnable:
        query_enhancer_chain = (
            RunnablePassthrough().assign(
                question=lambda x: clean_question(x["query"]),
            )
            | RunnableParallel(
                question=itemgetter("question"),
                standalone_question=self.question_condenser.chain,
                language=self.language_detector.chain,
                chat_history=RunnableLambda(
                    lambda x: convert_to_messages(x["chat_history"])
                )
                | RunnableLambda(lambda x: messages_to_dict(x)),
            )
            | self.intents_enhancer.chain
            | RunnableParallel(
                standalone_question=itemgetter("standalone_question"),
                language=itemgetter("language"),
                question=itemgetter("question"),
                intents=itemgetter("intents"),
                chat_history=itemgetter("chat_history"),
                keywords=self.keywords_enhancer.chain,
                vector_search=self.vector_search_enhancer.chain,
                web_results=self.web_search_enhancer.chain,
                avoid_query=itemgetter("avoid_query"),
            )
        )
        return query_enhancer_chain
