# llm-service-adapter/adapters/slack/adapter.py
import functools
from typing import Callable, Tuple
from adapters.base_adapter import BaseAdapter
import asyncio

class SlackAdapter(BaseAdapter):
    def __init__(self, llm_service, db_service, config: dict = {}):
        self.llm_service = llm_service
        self.db_service = db_service

    def process_query(self, query: str) -> Tuple[str, str, Tuple[float, float, float]]:
        query, response, timings = self.llm_service.chat(query)
        return query, response, timings

    async def run_chat(self, blocking_func: Callable, *args, **kwargs) -> Tuple[str, Tuple[float, float, float]]:
        func = functools.partial(
            blocking_func, *args, **kwargs
        )
        return await asyncio.get_event_loop().run_in_executor(None, func)

    async def add_response_to_db(self, user_id, run_id, query, response, feedback, elapsed_time, start_time):
        self.db_service.add_response(
            "slack", user_id, run_id, query, response, feedback, elapsed_time, start_time
        )
