from adapters.discord.adapter import DiscordAdapter
# from adapters.slack.adapter import SlackAdapter


class AdapterManager:
    def get_adapter(self, adapter_name, llm_service, db_service, config: dict = {}):
        if adapter_name == "discord":
            return DiscordAdapter(llm_service=llm_service, db_service=db_service)
        # elif adapter_name == "slack":
        #     return SlackAdapter(llm_service=llm_service, db_service=db_service)
        else:
            raise ValueError(f"Invalid service name: {adapter_name}")