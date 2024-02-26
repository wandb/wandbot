import re
import logging
from typing import Any, Dict, List

from slack_sdk.web.async_client import AsyncWebClient
from wandbot.api.client import AsyncAPIClient
from wandbot.apps.slack.utils import get_initial_message_from_thread


def get_adcopy_init_blocks() -> List[Dict[str, Any]]:
    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "What kind of ad copy are you looking for ?",
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
                "text": "*Executive Awareness*\n More clicks among Non-Technical audience",
            },
            "accessory": {
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": "Executive Awareness",
                },
                "value": "executive_awareness",
                "action_id": "executive_awareness",
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*Executive Sign-ups*\n Better conversion among Non-Technical audience",
            },
            "accessory": {
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": "Executive Sign-Ups",
                },
                "value": "executive_signups",
                "action_id": "executive_signups",
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*Practitioner Awareness*\n More clicks among Technical audience",
            },
            "accessory": {
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": "Technical Awareness",
                },
                "value": "technical_awareness",
                "action_id": "technical_awareness",
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*Practitioner Sign-Ups*\n Better conversion among Technical audience",
            },
            "accessory": {
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": "Technical Sign-Ups",
                },
                "value": "technical_signups",
                "action_id": "technical_signups",
            },
        },
    ]
    return blocks


def create_adcopy_init_handler(slack_client: AsyncWebClient) -> callable:
    async def adcopy_init_handler(
        ack: callable, body: dict, say: callable, logger: logging.Logger
    ) -> None:
        await ack()

        logger.info(f"Received message: {body}")
        initial_message = await get_initial_message_from_thread(
            slack_client, body["message"], body["channel"].get("id")
        )
        logger.info(f"Initial message: {initial_message}")

        thread_ts = initial_message.get(
            "thread_ts", None
        ) or initial_message.get("ts", None)

        adcopy_blocks = get_adcopy_init_blocks()
        await say(blocks=adcopy_blocks, thread_ts=thread_ts)

    return adcopy_init_handler


async def handle_adcopy_action(
    slack_client: AsyncWebClient,
    api_client: AsyncAPIClient,
    body: dict,
    say: callable,
    action: str,
    persona: str,
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
    query = re.sub(r"\<@\w+\>", "", query).strip()
    logger.info(f"Initial message: {initial_message}")

    await say(f"Working on generating ads for '{persona}' focussed on '{action}' \
for the query: '{query}'...", thread_ts=thread_ts)

    api_response = await api_client.generate_ads(
        query=query, action=action, persona=persona
    )

    await say(api_response.ad_copies, thread_ts=thread_ts)


def create_executive_awareness_handler(
    slack_client: AsyncWebClient, api_client: AsyncAPIClient
) -> callable:
    async def executive_awareness_handler(
        ack: callable, body: dict, say: callable, logger: logging.Logger
    ) -> None:
        await ack()
        await handle_adcopy_action(
            slack_client=slack_client,
            api_client=api_client,
            body=body,
            say=say,
            action="awareness",
            persona="executive",
            logger=logger,
        )

    return executive_awareness_handler


def create_executive_signups_handler(
    slack_client: AsyncWebClient, api_client: AsyncAPIClient
) -> callable:
    async def executive_signups_handler(
        ack: callable, body: dict, say: callable, logger: logging.Logger
    ) -> None:
        await ack()
        await handle_adcopy_action(
            slack_client=slack_client,
            api_client=api_client,
            body=body,
            say=say,
            action="signups",
            persona="executive",
            logger=logger,
        )

    return executive_signups_handler


def create_technical_awareness_handler(
    slack_client: AsyncWebClient, api_client: AsyncAPIClient
) -> callable:
    async def technical_awareness_handler(
        ack: callable, body: dict, say: callable, logger: logging.Logger
    ) -> None:
        await ack()
        await handle_adcopy_action(
            slack_client=slack_client,
            api_client=api_client,
            body=body,
            say=say,
            action="awareness",
            persona="technical",
            logger=logger,
        )

    return technical_awareness_handler


def create_technical_signups_handler(
    slack_client: AsyncWebClient, api_client: AsyncAPIClient
) -> callable:
    async def technical_signups_handler(
        ack: callable, body: dict, say: callable, logger: logging.Logger
    ) -> None:
        await ack()
        await handle_adcopy_action(
            slack_client=slack_client,
            api_client=api_client,
            body=body,
            say=say,
            action="signups",
            persona="technical",
            logger=logger,
        )

    return technical_signups_handler
