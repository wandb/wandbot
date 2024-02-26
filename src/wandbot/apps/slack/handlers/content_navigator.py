import logging

from slack_sdk.web.async_client import AsyncWebClient
from wandbot.api.client import AsyncAPIClient
from wandbot.apps.slack.utils import get_initial_message_from_thread


async def handle_content_navigator_action(
    slack_client: AsyncWebClient,
    api_client: AsyncAPIClient,
    body: dict,
    say: callable,
    logger: logging.Logger,
) -> None:
    logger.info(f"Received message: {body}")
    initial_message = await get_initial_message_from_thread(
        slack_client, body["message"], body["channel"].get("id")
    )
    thread_ts = initial_message.get("thread_ts", None) or initial_message.get(
        "ts", None
    )
    query = initial_message.get("text")
    user_id = initial_message.get("user", "")

    logger.info(f"Initial message: {initial_message}")
    await say("Working on it...", thread_ts=thread_ts)
    api_response = await api_client.generate_content_suggestions(
        query=query, user_id=user_id
    )

    # if there are any content suggestions, send them
    if api_response.response_items_count > 0:
        await say(api_response.slack_response, thread_ts=thread_ts)
    else:
        await say("No content suggestions found. Try rephrasing your query, but note \
there may also not be any relevant pieces of content for this query. Add '--debug' to \
your query and try again to see a detailed resoning for each suggestion.",
                thread_ts=thread_ts)
    
    # if debug mode is enabled, send the rejected suggestions as well
    if len(api_response.rejected_slack_response) > 1:
        await say("REJECTED SUGGESTIONS:\n{api_response.rejected_slack_response}", thread_ts=thread_ts)
        

def create_content_navigator_handler(
    slack_client: AsyncWebClient, api_client: AsyncAPIClient
) -> callable:
    async def executive_signups_handler(
        ack: callable, body: dict, say: callable, logger: logging.Logger
    ) -> None:
        await ack()
        await handle_content_navigator_action(
            slack_client=slack_client,
            api_client=api_client,
            body=body,
            say=say,
            logger=logger,
        )

    return executive_signups_handler