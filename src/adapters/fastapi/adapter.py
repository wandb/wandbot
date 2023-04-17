# llm-service-adapter/adapters/fastapi/adapter.py
from typing import Tuple
from adapters.base_adapter import BaseAdapter

class FastAPIAdapter(BaseAdapter):
    def __init__(self, llm_service, db_service, config: dict = {}):
        self.llm_service = llm_service
        self.db_service = db_service

    def process_query(self, query: str) -> Tuple[str, str, Tuple[float, float, float]]:
        query, response, timings = self.llm_service.chat(query)
        return query, response, timings

    def add_response_to_db(self, user_id, run_id, query, response, feedback, elapsed_time, start_time):
        self.db_service.add_response(
            "fastapi", user_id, run_id, query, response, feedback, elapsed_time, start_time
        )