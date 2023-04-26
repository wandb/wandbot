import aiohttp
import requests
from pydantic import AnyHttpUrl
from wandbot.api.schemas import APIFeedbackRequest, APIQueryRequest, APIQueryResponse


class APIClient:
    def __init__(self, url: AnyHttpUrl):
        self.url = url
        self.query_endpoint = f"{self.url}/query"
        self.feedback_endpoint = f"{self.url}/feedback"

    def query(self, request: APIQueryRequest) -> APIQueryResponse | None:
        with requests.Session() as session:
            with session.post(self.query_endpoint, json=request.dict()) as response:
                if response.status_code == 200:
                    response = response.json()
                    return APIQueryResponse(**response)
                else:
                    return None

    def feedback(self, request: APIFeedbackRequest) -> bool:
        with requests.Session() as session:
            with session.post(self.feedback_endpoint, json=request.dict()) as response:
                if response.status_code == 201:
                    return True
                else:
                    return False


class AsyncAPIClient(APIClient):
    def __init__(self, url: AnyHttpUrl):
        super().__init__(url)

    async def query(self, request: APIQueryRequest) -> APIQueryResponse | None:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.query_endpoint, json=request.dict()
            ) as response:
                if response.status == 200:
                    response = await response.json()
                    return APIQueryResponse(**response)
                else:
                    return None

    async def feedback(self, request: APIFeedbackRequest) -> bool:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.query_endpoint, json=request.dict()
            ) as response:
                if response.status == 201:
                    return True
                else:
                    return False
