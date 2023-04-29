import json
import uuid
from typing import List

import requests
from wandbot.api.schemas import (
    APIFeedbackRequest,
    APIGetChatThreadRequest,
    APIGetChatThreadResponse,
    APIPostChatThreadRequest,
    APIQueryRequest,
    APIQueryResponse,
)
from wandbot.database.schemas import QuestionAnswer


class APIClient:
    def __init__(self, url: str):
        self.url = url
        self.query_endpoint = f"{self.url}/query"
        self.feedback_endpoint = f"{self.url}/feedback"
        self.chat_thread_endpoint = f"{self.url}/chat_thread"

    def _get_chat_thread(
        self, request: APIGetChatThreadRequest
    ) -> APIGetChatThreadResponse | None:
        with requests.Session() as session:
            with session.get(
                f"{self.chat_thread_endpoint}/{request.thread_id}"
            ) as response:
                if response.status_code == 200:
                    response = response.json()
                    return APIGetChatThreadResponse(**response)
                else:
                    return None

    def get_chat_history(
        self,
        thread_id: str,
    ) -> List[QuestionAnswer] | None:
        request = APIGetChatThreadRequest(
            thread_id=thread_id,
        )
        response = self._get_chat_thread(request)
        if response is None:
            return None
        else:
            return response.question_answers

    def _create_chat_thread(self, request: APIPostChatThreadRequest) -> bool:
        with requests.Session() as session:
            with session.post(
                self.chat_thread_endpoint, json=json.loads(request.json())
            ) as response:
                if response.status_code == 201:
                    return True
                else:
                    return False

    def save_chat_history(
        self,
        thread_id: str,
        application: str | None,
        chat_history: List[QuestionAnswer],
    ) -> bool:
        request = APIPostChatThreadRequest(
            thread_id=thread_id,
            application=application,
            question_answers=chat_history,
        )
        response = self._create_chat_thread(request)
        return response

    def _query(self, request: APIQueryRequest) -> APIQueryResponse | None:
        with requests.Session() as session:
            payload = json.loads(request.json())
            with session.post(self.query_endpoint, json=payload) as response:
                if response.status_code == 200:
                    response = response.json()
                    return APIQueryResponse(**response)
                else:
                    return None

    def query(
        self,
        thread_id: str,
        query: str,
        chat_history: List[QuestionAnswer],
    ) -> APIQueryResponse:
        request = APIQueryRequest(
            thread_id=thread_id,
            question=query,
            chat_history=chat_history,
        )
        response = self._query(request)

        return response

    def feedback(self, request: APIFeedbackRequest) -> bool:
        with requests.Session() as session:
            with session.post(self.feedback_endpoint, json=request.dict()) as response:
                if response.status_code == 201:
                    return True
                else:
                    return False


# class AsyncAPIClient(APIClient):
#     def __init__(self, url: AnyHttpUrl):
#         super().__init__(url)
#
#     async def query(self, request: APIQueryRequest) -> APIQueryResponse | None:
#         async with aiohttp.ClientSession() as session:
#             async with session.post(
#                 self.query_endpoint, json=request.dict()
#             ) as response:
#                 if response.status == 200:
#                     response = await response.json()
#                     return APIQueryResponse(**response)
#                 else:
#                     return None
#
#     async def feedback(self, request: APIFeedbackRequest) -> bool:
#         async with aiohttp.ClientSession() as session:
#             async with session.post(
#                 self.query_endpoint, json=request.dict()
#             ) as response:
#                 if response.status == 201:
#                     return True
#                 else:
#                     return False
#
#     async def get_thread(
#         self, request: APIGetThreadRequest
#     ) -> APIGetThreadResponse | None:
#         async with aiohttp.ClientSession() as session:
#             async with session.get(
#                 self.chat_thread_endpoint, json=request.dict()
#             ) as response:
#                 if response.status == 200:
#                     response = await response.json()
#                     return APIGetThreadResponse(**response)
#                 else:
#                     return None


if __name__ == "__main__":
    api_client = APIClient(url="http://localhost:8000")

    thread_id = "5ff6e13f-0a44-423c-b897-dc171c35c952"
    query = "Hello, this is a new question"
    thread = api_client.get_chat_history(thread_id=thread_id)
    chat_history = api_client.get_chat_history(thread_id=thread_id)

    if not chat_history:
        print("No chat history found")
    else:
        print(json.dumps([json.loads(item.json()) for item in chat_history], indent=2))

    response = api_client.query(
        thread_id=thread_id, query=query, chat_history=chat_history
    )
    print(response.json(indent=2))
    saved_response = api_client.save_chat_history(
        thread_id=thread_id,
        application="test",
        chat_history=[
            QuestionAnswer(**response.dict(), question_answer_id=str(uuid.uuid4()))
        ],
    )
    print(saved_response)
