from managers import ServiceManager, AdapterManager
from adapters.discord.app import create_bot
import os

selected_service = "wandbot"
selected_app = "discord"

#To load in your custom chat application
service_manager = ServiceManager()
llm_service = service_manager.get_llm_service(selected_service)
db_service = service_manager.get_db_service()

#To allow our bots access to the chat application
adapter_manager = AdapterManager()
discord_adapter = adapter_manager.get_adapter(selected_app, llm_service, db_service)

if selected_app == "discord":
    bot = create_bot(discord_adapter)
    bot.run(os.environ.get("DISCORD_BOT_TOKEN"))