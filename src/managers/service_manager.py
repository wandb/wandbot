# llm-service-adapter/managers/service_manager.py
from services.wandbot.llm_service import WandbotLLMService
from services.db_service import DBService

class ServiceManager:
    def __init__(self):
        self.llm_services = {"wandbot": WandbotLLMService}
        self.db_services = {"default": DBService}

    def get_llm_service(self, name: str):
        return self.llm_services[name]()

    def register_llm_service(self, name: str, service_class):
        self.llm_services[name] = service_class

    def get_db_service(self):
        return self.db_services["default"]()