"""A Slack bot that interacts with users and processes their queries.

This module contains the main functionality of the Slack bot. It listens for mentions of the bot in messages,
processes the text of the message, and sends a response. It also handles reactions added to messages and 
saves them as feedback. The bot supports both English and Japanese languages.

The bot uses the Slack Bolt framework for handling events and the langdetect library for language detection.
It also communicates with an external API for processing queries and storing chat history and feedback.

"""

import argparse
import asyncio
from typing import Any, Dict, List

from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp
from wandbot.api.client import AsyncAPIClient
from wandbot.apps.slack.config import SlackAppEnConfig, SlackAppJaConfig
from wandbot.apps.slack.handlers.ad_copy import (
    create_adcopy_init_handler,
    create_executive_awareness_handler,
    create_executive_signups_handler,
    create_technical_awareness_handler,
    create_technical_signups_handler,
)
from wandbot.apps.slack.handlers.content_navigator import (
    create_content_navigator_handler,
)
from wandbot.apps.slack.handlers.docsbot import (
    create_docsbot_handler,
    create_reaction_added_handler,
)
from wandbot.utils import get_logger

logger = get_logger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument(
    "-l",
    "--language",
    default="en",
    help="Language of the bot",
    type=str,
    choices=["en", "ja"],
)

args = parser.parse_args()

if args.language == "ja":
    config = SlackAppJaConfig()
else:
    config = SlackAppEnConfig()


app = AsyncApp(token=config.SLACK_BOT_TOKEN)
api_client = AsyncAPIClient(url=config.WANDBOT_API_URL)
slack_client = app.client


def get_init_block(user: str) -> List[Dict[str, Any]]:
    initial_block = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "Hi <@{user}>, please confirm the action to take ".format(
                    user=user
                ),
            },
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*Options:*"},
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*Get W&B Technical Support*\n Technical support for the W&B app and SDK",
            },
            "accessory": {
                "type": "button",
                "text": {"type": "plain_text", "text": "Support Bot"},
                "value": "docsbot",
                "action_id": "docsbot",
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*Create W&B Ad Copy*\n Create ad-copy suggestions, optimized for W&B",
            },
            "accessory": {
                "type": "button",
                "text": {"type": "plain_text", "text": "Ad Copy"},
                "value": "adcopy",
                "action_id": "adcopy",
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*Suggest W&B Content*\n Suggest W&B articles, whitepapers, courses etc related to your query",
            },
            "accessory": {
                "type": "button",
                "text": {"type": "plain_text", "text": "Content Navigator"},
                "value": "content_navigator",
                "action_id": "content_navigator",
            },
        },
    ]
    return initial_block


# --------------------------------------
# Main Wandbot Mention Handler
# --------------------------------------
@app.event("app_mention")
async def handle_mention(event: dict, say: callable) -> None:
    original_msg_ts = event.get("ts")
    user = event.get("user")
    init_block = get_init_block(user=user)
    await say(blocks=init_block, thread_ts=original_msg_ts)


# --------------------------------------
# Handlers for the Docsbot
# --------------------------------------
app.action("docsbot")(
    create_docsbot_handler(
        config=config, slack_client=slack_client, api_client=api_client
    )
)

app.event("reaction_added")(
    create_reaction_added_handler(
        slack_client=slack_client, api_client=api_client
    )
)


# --------------------------------------
# Handlers for the Ad Copy Generator
# --------------------------------------

app.action("adcopy")(create_adcopy_init_handler(slack_client=slack_client))

app.action("executive_awareness")(
    create_executive_awareness_handler(
        slack_client=slack_client, api_client=api_client
    )
)

app.action("executive_signups")(
    create_executive_signups_handler(
        slack_client=slack_client, api_client=api_client
    )
)

app.action("technical_awareness")(
    create_technical_awareness_handler(
        slack_client=slack_client, api_client=api_client
    )
)

app.action("technical_signups")(
    create_technical_signups_handler(
        slack_client=slack_client, api_client=api_client
    )
)


# --------------------------------------
# Handlers for the Content Navigator
# --------------------------------------

app.action("content_navigator")(
    create_content_navigator_handler(
        slack_client=slack_client, api_client=api_client
    )
)


async def main():
    handler = AsyncSocketModeHandler(app, config.SLACK_APP_TOKEN)
    await handler.start_async()


if __name__ == "__main__":
    asyncio.run(main())
