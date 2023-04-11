from services.db_service import DBService
from services.wandbot.llm_service import WandbotLLMService
# from services.github_reader.llm_service import GithubReaderLLMService

class ServiceManager:
    def get_db_service(self, config: dict = {}):
        return DBService(config=config)

    def get_llm_service(self, service_name, config: dict = {}):
        if service_name == "wandbot":
            return WandbotLLMService(config=config)
        # elif service_name == "github-reader":
        #     return GithubReaderLLMService()
        else:
            raise ValueError(f"Invalid service name: {service_name}")