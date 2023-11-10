"""A module for API schemas.

This module provides the schemas for API requests and responses.
It includes classes for creating question answers, getting chat threads,
creating chat threads, querying, creating feedback, and more.

Classes:
    APIQuestionAnswerRequest: Request schema for creating a question answer.
    APIQuestionAnswerResponse: Response schema for a question answer.
    APIGetChatThreadRequest: Request schema for getting a chat thread.
    APIGetChatThreadResponse: Response schema for a chat thread.
    APICreateChatThreadRequest: Request schema for creating a chat thread.
    APIQueryRequest: Request schema for querying.
    APIQueryResponse: Response schema for a query.
    APIFeedbackRequest: Request schema for creating feedback.
    APIFeedbackResponse: Response schema for feedback.
"""

from wandbot.chat.schemas import ChatRepsonse, ChatRequest
from wandbot.database.schemas import (
    ChatThread,
    ChatThreadCreate,
    Feedback,
    FeedbackCreate,
    QuestionAnswer,
    QuestionAnswerCreate,
)


class APIQuestionAnswerRequest(QuestionAnswerCreate):
    pass


class APIQuestionAnswerResponse(QuestionAnswer):
    pass


class APIGetChatThreadRequest(ChatThreadCreate):
    pass


class APIGetChatThreadResponse(ChatThread):
    pass


class APICreateChatThreadRequest(ChatThreadCreate):
    pass


class APIQueryRequest(ChatRequest):
    pass


class APIQueryResponse(ChatRepsonse):
    pass


class APIFeedbackRequest(FeedbackCreate):
    pass


class APIFeedbackResponse(Feedback):
    pass
