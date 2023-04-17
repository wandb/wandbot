# llm-service-adapter/managers/adapter_manager.py
from adapters.discord.adapter import DiscordAdapter
from adapters.slack.adapter import SlackAdapter
from adapters.fastapi.adapter import FastAPIAdapter

class AdapterManager:
    def __init__(self):
        self.adapters = {
            "discord": DiscordAdapter,
            "slack": SlackAdapter,
            "fastapi": FastAPIAdapter
        }

    def get_adapter(self, name: str, llm_service, db_service):
        return self.adapters[name](llm_service, db_service)

    def register_adapter(self, name: str, adapter_class):
        self.adapters[name] = adapter_class