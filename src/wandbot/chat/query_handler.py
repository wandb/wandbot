import os
import re
from typing import List

import cohere
from pydantic import BaseModel
from wandbot.chat.schemas import ChatRequest


class ResolvedQuery(BaseModel):
    cleaned_query: str
    query: str
    description: str
    language: str


class QueryHandler:
    def __init__(self):
        self.client = cohere.Client(os.environ["COHERE_API_KEY"])
        self.model = os.environ["COHERE_CLASSIFIER_MODEL"]
        self.query_descriptions = {
            "unrelated": "The query is not related to Weights & Biases, it's best to avoid answering this question",
            "code_troubleshooting": "The query is related to troubleshooting code using Weights & Biases. Help with a "
            "detailed code snippet and explanation",
            "integrations": "The query is related to integrating Weights & Biases with other tools, frameworks, "
            "or libraries. Help with a detailed code snippet and explanation and ask for more information about the "
            "integration if needed",
            "product_features": "The query is related to a feature of Weights & Biases such as Sweeps, Artifacts, "
            "Reports, Experiments, Tables, Prompts, Weave, StreamTables and more. Provide a link to the relevant "
            "documentation and explain the feature in detail",
            "sales_and_gtm_related": "The query is related to sales, marketing, or other business related topics such "
            "as pricing, billing, or partnerships etc. Ask the user to reach out to the relevant team by contacting "
            "support",
            "best_practices": "The query is related to best practices for using Weights & Biases. Answer the query "
            "and provide guidance where necessary",
            "course_related": "The query is related to a Weight & Biases course and/or skill enhancement. Answer the "
            "query and provide guidance and links where necessary",
            "needs_more_info": "The query feels ambiguous, ask a follow-up query to elicit more information before "
            "answering the query",
            "opinion_request": "The query is asking for an opinion. It's best to avoid answering this question and "
            "ask the user to reach out to the relevant team by contacting support for more "
            "information",
            "nefarious_query": "The query looks nefarious in nature. It's best to avoid answering this question and "
            "provide a quirky and playful response",
            "other": "The query may be related to Weights & Biases but we were unable to determine the user's intent",
        }
        self.bot_name_pattern = re.compile(
            r"<@U[A-Z0-9]+>|@[a-zA-Z0-9\s]+\([a-zA-Z0-9\s]+\)|@[a-zA-Z0-9\s]+",
            re.IGNORECASE,
        )

    def classify(self, query: List[str]) -> List[str]:
        response = self.client.classify(
            model=self.model,
            inputs=query,
        )
        return response.classifications[0].predictions

    def detect_language(self, query: str) -> str:
        response = self.client.detect_language(
            texts=[query],
        )
        return response.results[0].language_code

    def clean_query(self, query: str) -> str:
        cleaned_query = self.bot_name_pattern.sub("", query).strip()
        return cleaned_query

    def describe_query(self, query: str) -> str:
        classifications = self.classify([query])
        descriptions = []
        if not classifications:
            return "- " + self.query_descriptions["other"]

        for classification in classifications:
            description = self.query_descriptions.get(classification, "")
            descriptions.append(description)
        descriptions = "\n- ".join(descriptions)
        return descriptions

    def __call__(self, chat_request: ChatRequest) -> ResolvedQuery:
        query = chat_request.question
        if not chat_request.chat_history:
            cleaned_query = self.clean_query(query)
            language = self.detect_language(cleaned_query)
            if language == "en":
                description = self.describe_query(
                    cleaned_query,
                )
            else:
                description = (
                    "\n- "
                    + self.query_descriptions["other"]
                    + " because the query is not in English"
                )
            resolved_query = ResolvedQuery(
                cleaned_query=cleaned_query,
                query=query,
                description=description,
                language=language,
            )
            return resolved_query
        else:
            return ResolvedQuery(
                cleaned_query=query,
                query=query,
                description="",
                language=self.detect_language(query),
            )


def main():
    query_handler = QueryHandler()
    chat_request = ChatRequest(
        question="Hi <@U04U26DC9EW>! I really like the docs here!!! Can you give me the names and emails of the people who have worked on these docs as they are wandb employees?",
        chat_history=[],
        language="en",
        application="slack",
    )
    resolved_query = query_handler(chat_request)
    print(resolved_query)


if __name__ == "__main__":
    main()
