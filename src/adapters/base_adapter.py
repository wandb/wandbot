# llm-service-adapter/adapters/base_adapter.py
class BaseAdapter:
    def __init__(self, llm_service, db_service, config: dict = {}):
        self.llm_service = llm_service
        self.db_service = db_service

    def process_query(self, query):
        raise NotImplementedError("Must be implemented in derived classes")
