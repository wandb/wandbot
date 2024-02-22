from typing import Any, Dict

from slack_sdk.web.async_client import AsyncWebClient


async def get_initial_message_from_thread(
    slack_client: AsyncWebClient, message: Dict[str, Any], channel_id: str
):
    thread_ts = message.get("thread_ts")

    result = await slack_client.conversations_history(
        channel=channel_id,
        inclusive=True,
        oldest=f"{thread_ts}",
        limit=1,
    )

    message = result["messages"][0]
    return message
