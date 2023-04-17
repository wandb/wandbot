# llm-service-adapter/main.py
import argparse
from managers import ServiceManager, AdapterManager
import os

# Set up the command line arguments
parser = argparse.ArgumentParser(description="Run the chatbot for Discord or Slack.")
parser.add_argument("--service", default="wandbot", help="The service to use (default: wandbot).")
parser.add_argument("--app", default="discord", help="The app to use (default: discord).")
args = parser.parse_args()

selected_service = args.service
selected_app = args.app

# To load in your custom chat application
service_manager = ServiceManager()
llm_service = service_manager.get_llm_service(selected_service)
db_service = service_manager.get_db_service()

# To allow our bots access to the chat application
adapter_manager = AdapterManager()
adapter = adapter_manager.get_adapter(selected_app, llm_service, db_service)

if selected_app == "discord":
    from adapters.discord.app import create_bot
    bot = create_bot(adapter)
    bot.run(os.environ.get("DISCORD_BOT_TOKEN"))
elif selected_app == "slack":
    from adapters.slack.app import create_app
    from slack_bolt.adapter.socket_mode import SocketModeHandler
    app = create_app(adapter)
    handler = SocketModeHandler(app, os.environ.get("SLACK_APP_TOKEN"))
    handler.start()
elif selected_app == "fastapi":
    from adapters.fastapi.app import create_app
    import uvicorn
    app = create_app(adapter)
    uvicorn.run(app, host="0.0.0.0", port=8000) 