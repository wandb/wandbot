import json
import uuid
from datetime import datetime
from typing import List

import aiohttp
import requests
from wandbot.api.schemas import (
    APIFeedbackRequest,
    APIFeedbackResponse,
    APIGetChatThreadRequest,
    APIGetChatThreadResponse,
    APIQueryRequest,
    APIQueryResponse,
    APIQuestionAnswerRequest,
    APIQuestionAnswerResponse,
)
from wandbot.database.schemas import QuestionAnswer


class APIClient:
    def __init__(self, url: str):
        self.url = url
        self.query_endpoint = f"{self.url}/query"
        self.feedback_endpoint = f"{self.url}/feedback"
        self.chat_thread_endpoint = f"{self.url}/chat_thread"
        self.chat_question_answer_endpoint = f"{self.url}/question_answer"

    def _get_chat_thread(
        self, request: APIGetChatThreadRequest
    ) -> APIGetChatThreadResponse | None:
        with requests.Session() as session:
            with session.get(
                f"{self.chat_thread_endpoint}/{request.application}/{request.thread_id}"
            ) as response:
                if response.status_code in (200, 201):
                    return APIGetChatThreadResponse(**response.json())

    def get_chat_history(
        self,
        application: str,
        thread_id: str,
    ) -> List[QuestionAnswer] | None:
        request = APIGetChatThreadRequest(
            application=application,
            thread_id=thread_id,
        )
        response = self._get_chat_thread(request)
        if response is None:
            return None
        else:
            return response.question_answers

    def _create_question_answer(
        self, request: APIQuestionAnswerRequest
    ) -> APIQuestionAnswerResponse | None:
        with requests.Session() as session:
            with session.post(
                self.chat_question_answer_endpoint, json=json.loads(request.json())
            ) as response:
                if response.status_code == 201:
                    return APIQuestionAnswerResponse(**response.json())

    def create_question_answer(
        self,
        question_answer_id: str,
        thread_id: str,
        question: str,
        answer: str | None = None,
        model: str | None = None,
        sources: str | None = None,
        source_documents: str | None = None,
        total_tokens: int | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        successful_requests: int | None = None,
        total_cost: float | None = None,
        time_taken: float | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> APIQuestionAnswerResponse | None:
        request = APIQuestionAnswerRequest(
            question_answer_id=question_answer_id,
            thread_id=thread_id,
            question=question,
            answer=answer,
            model=model,
            sources=sources,
            source_documents=source_documents,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            successful_requests=successful_requests,
            total_cost=total_cost,
            time_taken=time_taken,
            start_time=start_time,
            end_time=end_time,
        )
        response = self._create_question_answer(request)
        return response

    def _create_feedback(
        self, request: APIFeedbackRequest
    ) -> APIFeedbackResponse | None:
        with requests.Session() as session:
            with session.post(self.feedback_endpoint, json=request.dict()) as response:
                if response.status_code == 201:
                    return APIFeedbackResponse(**response.json())

    def create_feedback(self, feedback_id: str, question_answer_id: str, rating: int):
        feedback_request = APIFeedbackRequest(
            feedback_id=feedback_id,
            question_answer_id=question_answer_id,
            rating=rating,
        )
        response = self._create_feedback(feedback_request)
        return response

    def _query(self, request: APIQueryRequest) -> APIQueryResponse | None:
        with requests.Session() as session:
            payload = json.loads(request.json())
            with session.post(self.query_endpoint, json=payload) as response:
                if response.status_code == 200:
                    return APIQueryResponse(**response.json())
                else:
                    return None

    def query(
        self,
        question: str,
        chat_history: List[QuestionAnswer],
    ) -> APIQueryResponse:
        request = APIQueryRequest(
            question=question,
            chat_history=chat_history,
        )
        response = self._query(request)

        return response


class AsyncAPIClient(APIClient):
    def __init__(self, url: str):
        super().__init__(url)

    async def _get_chat_thread(
        self, request: APIGetChatThreadRequest
    ) -> APIGetChatThreadResponse | None:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.chat_thread_endpoint}/{request.application}/{request.thread_id}"
            ) as response:
                if response.status in (200, 201):
                    response = await response.json()
                    return APIGetChatThreadResponse(**response)

    async def get_chat_history(
        self, application: str, thread_id: str
    ) -> List[QuestionAnswer] | None:
        request = APIGetChatThreadRequest(
            application=application,
            thread_id=thread_id,
        )
        response = await self._get_chat_thread(request)
        if response is None:
            return None
        else:
            return response.question_answers

    async def _create_question_answer(
        self, request: APIQuestionAnswerRequest
    ) -> APIQuestionAnswerResponse | None:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.chat_question_answer_endpoint, json=json.loads(request.json())
            ) as response:
                if response.status == 201:
                    response = await response.json()
                    return APIQuestionAnswerResponse(**response)

    async def create_question_answer(
        self,
        question_answer_id: str,
        thread_id: str,
        question: str,
        answer: str | None = None,
        model: str | None = None,
        sources: str | None = None,
        source_documents: str | None = None,
        total_tokens: int | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        successful_requests: int | None = None,
        total_cost: float | None = None,
        time_taken: float | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> APIQuestionAnswerResponse | None:
        request = APIQuestionAnswerRequest(
            question_answer_id=question_answer_id,
            thread_id=thread_id,
            question=question,
            answer=answer,
            model=model,
            sources=sources,
            source_documents=source_documents,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            successful_requests=successful_requests,
            total_cost=total_cost,
            time_taken=time_taken,
            start_time=start_time,
            end_time=end_time,
        )
        response = await self._create_question_answer(request)
        return response

    async def _create_feedback(
        self, request: APIFeedbackRequest
    ) -> APIFeedbackResponse:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.feedback_endpoint, json=json.loads(request.json())
            ) as response:
                if response.status == 201:
                    response = await response.json()
                    return APIFeedbackResponse(**response)

    async def create_feedback(
        self, feedback_id: str, question_answer_id: str, rating: int
    ):
        request = APIFeedbackRequest(
            feedback_id=feedback_id,
            question_answer_id=question_answer_id,
            rating=rating,
        )
        response = await self._create_feedback(request)
        return response

    async def _query(self, request: APIQueryRequest) -> APIQueryResponse | None:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.query_endpoint, json=json.loads(request.json())
            ) as response:
                if response.status == 200:
                    response = await response.json()
                    return APIQueryResponse(**response)
                else:
                    return None

    async def query(
        self,
        question: str,
        chat_history: List[QuestionAnswer],
    ) -> APIQueryResponse:
        request = APIQueryRequest(
            question=question,
            chat_history=chat_history,
        )
        response = await self._query(request)

        return response


if __name__ == "__main__":
    import asyncio

    async def run():
        from wandbot.ingestion.utils import Timer

        with Timer() as timer:
            api_client = AsyncAPIClient(url="http://localhost:8000")

            application = "test"
            # thread_id = str(uuid.uuid4())
            thread_id = "300d9a8c-ea55-4bb1-94e6-d3e3ed2df8bd"
            chat_history = await api_client.get_chat_history(
                application=application, thread_id=thread_id
            )

            if not chat_history:
                print("No chat history found")
            else:
                print(
                    json.dumps(
                        [json.loads(item.json()) for item in chat_history], indent=2
                    )
                )
                # chat_history = [(item.question, item.answer) for item in chat_history]

            # query the api and get the chat response
            question = "Hi @wandbot, How about openai?"
            chat_response = await api_client.query(
                question=question, chat_history=chat_history
            )
            # save the chat response to the database
            question_answer_id = str(uuid.uuid4())

            await api_client.create_question_answer(
                question_answer_id=question_answer_id,
                thread_id=thread_id,
                **chat_response.dict(),
            )

            # get the chat history again
            chat_history = await api_client.get_chat_history(
                application=application, thread_id=thread_id
            )
            print(
                json.dumps([json.loads(item.json()) for item in chat_history], indent=2)
            )

            # add feedback
            feedback_id = str(uuid.uuid4())
            await api_client.create_feedback(
                feedback_id=feedback_id, question_answer_id=question_answer_id, rating=1
            )

            # get the chat history again
            chat_history = await api_client.get_chat_history(
                application=application, thread_id=thread_id
            )
            print(
                json.dumps([json.loads(item.json()) for item in chat_history], indent=2)
            )
        print(timer.start, timer.start, timer.elapsed)

    asyncio.run(run())
