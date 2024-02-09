from operator import itemgetter

import regex as re
from langchain_core.runnables import (
    Runnable,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI

from wandbot.query_handler.history_handler import load_query_condense_chain
from wandbot.query_handler.intents_enhancer import load_intent_enhancement_chain
from wandbot.query_handler.keyword_search_enhancer import (
    load_keywords_enhancement_chain,
)
from wandbot.query_handler.language_detection import (
    load_language_detection_chain,
)
from wandbot.query_handler.vector_search_enhancer import (
    load_vectorsearch_enhancement_chain,
)
from wandbot.query_handler.web_search import load_web_answer_enhancement_chain

BOT_NAME_PATTERN = re.compile(r"<@U[A-Z0-9]+>|@[a-zA-Z0-9]+")


def clean_question(question: str) -> str:
    cleaned_query = BOT_NAME_PATTERN.sub("", question).strip()
    return cleaned_query


def load_query_enhancement_chain(
    model: ChatOpenAI, lang_detect_model_path: str
) -> Runnable:
    condense_question_chain = load_query_condense_chain(model)
    intent_enhancement_chain = load_intent_enhancement_chain(model)

    language_enhancement_chain = load_language_detection_chain(
        model=lang_detect_model_path
    )

    keywords_enhancement_chain = load_keywords_enhancement_chain(model)
    vector_search_enhancement_chain = load_vectorsearch_enhancement_chain(model)
    web_answer_enhancement_chain = load_web_answer_enhancement_chain(top_k=5)

    query_enhancer_chain = (
        RunnablePassthrough().assign(
            question=lambda x: clean_question(x["query"]),
        )
        | RunnableParallel(
            question=itemgetter("question"),
            standalone_question=condense_question_chain,
            language=language_enhancement_chain,
            chat_history=itemgetter("chat_history"),
        )
        | intent_enhancement_chain
        | RunnableParallel(
            standalone_question=itemgetter("standalone_question"),
            language=itemgetter("language"),
            question=itemgetter("question"),
            intents=itemgetter("intents"),
            chat_history=itemgetter("chat_history"),
            keywords=keywords_enhancement_chain,
            vector_search=vector_search_enhancement_chain,
            web_results=web_answer_enhancement_chain,
            avoid_query=itemgetter("avoid_query"),
        )
    )
    return query_enhancer_chain
