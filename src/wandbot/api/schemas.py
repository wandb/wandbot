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
