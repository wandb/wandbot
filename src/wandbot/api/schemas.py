from typing import List

from pydantic import BaseModel
from wandbot.config import ChatRepsonse
from wandbot.database.schemas import ChatThread, Feedback, QuestionAnswer


class APIGetChatThreadRequest(BaseModel):
    thread_id: str


class APIGetChatThreadResponse(ChatThread):
    pass


class APIPostChatThreadRequest(ChatThread):
    pass


class APIPostChatThreadResponse(ChatThread):
    pass


class APIQueryRequest(BaseModel):
    thread_id: str
    question: str
    chat_history: List[QuestionAnswer] | None = []


class APIQueryResponse(ChatRepsonse):
    thread_id: str
    question: str


class APIFeedbackRequest(Feedback):
    pass


class APIFeedbackResponse(QuestionAnswer):
    pass
