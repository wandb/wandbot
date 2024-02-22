import logging

from slack_sdk.web import SlackResponse
from slack_sdk.web.async_client import AsyncWebClient
from wandbot.api.client import AsyncAPIClient
from wandbot.apps.slack.config import SlackAppEnConfig, SlackAppJaConfig
from wandbot.apps.slack.formatter import MrkdwnFormatter
from wandbot.apps.slack.utils import get_initial_message_from_thread
from wandbot.apps.utils import format_response


async def send_message(
    say: callable, message: str, thread: str = None
) -> SlackResponse:
    message = MrkdwnFormatter()(message)
    if thread is not None:
        return await say(text=message, thread_ts=thread)
    else:
        return await say(text=message)


def create_docsbot_handler(
    config: SlackAppEnConfig | SlackAppJaConfig,
    slack_client: AsyncWebClient,
    api_client: AsyncAPIClient,
) -> callable:
    async def docsbot_handler(
        ack: callable, body: dict, say: callable, logger: logging.Logger
    ) -> None:
        """
        Handles the command when the app is mentioned in a message.

        Args:
            ack (function): The function to acknowledge the event.
            body (dict): The event body containing the message details.
            say (function): The function to send a message.
            logger (Logger): The logger instance for logging errors.

        Raises:
            Exception: If there is an error posting the message.
        """
        await ack()

        logger.info(f"Received message: {body}")
        initial_message = await get_initial_message_from_thread(
            slack_client,
            body["message"],
            body["channel"].get("id"),
        )
        logger.info(f"Initial message: {initial_message}")

        try:
            query = initial_message.get("text")
            user = initial_message.get("user")
            thread_id = initial_message.get(
                "thread_ts", None
            ) or initial_message.get("ts", None)
            chat_history = await api_client.get_chat_history(
                application=config.APPLICATION, thread_id=thread_id
            )

            if not chat_history:
                # send out the intro message
                await send_message(
                    say=say,
                    message=config.INTRO_MESSAGE.format(user=user),
                    thread=thread_id,
                )
            # process the query through the api
            api_response = await api_client.query(
                question=query,
                chat_history=chat_history,
                language=config.bot_language,
                application=config.APPLICATION,
            )
            response = format_response(
                config,
                api_response,
                config.OUTRO_MESSAGE,
            )

            # send the response
            sent_message = await send_message(
                say=say, message=response, thread=thread_id
            )

            await slack_client.reactions_add(
                channel=body["channel"].get("id"),
                timestamp=sent_message["ts"],
                name="thumbsup",
            )
            await slack_client.reactions_add(
                channel=body["channel"].get("id"),
                timestamp=sent_message["ts"],
                name="thumbsdown",
            )

            #  save the question answer to the database
            await api_client.create_question_answer(
                thread_id=thread_id,
                question_answer_id=sent_message["ts"],
                language=config.bot_language,
                **api_response.model_dump(),
            )

        except Exception as e:
            logger.error(f"Error posting message: {e}")

    return docsbot_handler


def parse_reaction(reaction: str) -> int:
    """
    Parses the reaction and returns the corresponding rating value.

    Args:
        reaction (str): The reaction emoji.

    Returns:
        int: The rating value (-1, 0, or 1).
    """
    if reaction == "+1":
        return 1
    elif reaction == "-1":
        return -1
    else:
        return 0


def create_reaction_added_handler(
    slack_client: AsyncWebClient, api_client: AsyncAPIClient
) -> callable:
    async def handle_reaction_added(event: dict) -> None:
        """
        Handles the event when a reaction is added to a message.

        Args:
            event (dict): The event details.

        """
        channel_id = event["item"]["channel"]
        message_ts = event["item"]["ts"]

        conversation = await slack_client.conversations_replies(
            channel=channel_id,
            ts=message_ts,
            inclusive=True,
            limit=1,
        )
        messages = conversation.get(
            "messages",
        )
        if messages and len(messages):
            thread_ts = messages[0].get("thread_ts")
            if thread_ts:
                rating = parse_reaction(event["reaction"])
                await api_client.create_feedback(
                    feedback_id=event["event_ts"],
                    question_answer_id=message_ts,
                    rating=rating,
                )

    return handle_reaction_added
