from llama_index import ChatPromptTemplate
from llama_index.llms import ChatMessage, MessageRole
from ragas.utils import json_loader


def make_eval_template(system_template, user_template):
    return ChatPromptTemplate(
        message_templates=[
            ChatMessage(role=MessageRole.SYSTEM, content=system_template),
            ChatMessage(role=MessageRole.USER, content=user_template),
        ]
    )


async def safe_parse_eval_response(eval_response, passing_decision):
    try:
        eval_response_dict = json_loader.safe_load(eval_response)
        score = eval_response_dict.get("score")
        reasoning = eval_response_dict.get("reason")
        decision = eval_response_dict.get("decision") == passing_decision

    except Exception:
        score = 0
        reasoning = "Unable to parse response"
        decision = False
    return decision, reasoning, score
