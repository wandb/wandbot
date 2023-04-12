from adapters.discord.adapter import DiscordAdapter
from adapters.slack.adapter import SlackAdapter
from adapters.fastapi.adapter import FastAPIAdapter

#Should i make this return both the adapter and app?
class AdapterManager:
    def get_adapter(self, adapter_name, llm_service, db_service, config: dict = {}):
        if adapter_name == "discord":
            return DiscordAdapter(llm_service=llm_service, db_service=db_service, config=config)
        elif adapter_name == "slack":
            return SlackAdapter(llm_service=llm_service, db_service=db_service, config=config)
        elif adapter_name == "fastapi":
            return FastAPIAdapter(llm_service=llm_service, db_service=db_service, config=config)
        else:
            raise ValueError(f"Invalid service name: {adapter_name}")