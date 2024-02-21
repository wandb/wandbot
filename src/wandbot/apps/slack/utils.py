from typing import Any, Dict, List


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
                "text": "*W&B Technical Support*\n Technical support for the W&B app and SDK",
            },
            "accessory": {
                "type": "button",
                "text": {"type": "plain_text", "text": "Docs Bot"},
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
                "text": "*Find Content*\n Suggest Fully Connected articles related to your query",
            },
            "accessory": {
                "type": "button",
                "text": {"type": "plain_text", "text": "Content Suggestor"},
                "value": "content_suggestor",
                "action_id": "content_suggestor",
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*Chat with a Youtube Video*\n Chat with the YouTube link provided",
            },
            "accessory": {
                "type": "button",
                "text": {"type": "plain_text", "text": "YT Assistant"},
                "value": "yt_assistant",
                "action_id": "yt_assistant",
            },
        },
    ]
    return initial_block


async def get_initial_message(app, message, channel_id, token):
    thread_ts = message.get("thread_ts")

    result = await app.client.conversations_history(
        channel=channel_id,
        inclusive=True,
        oldest=f"{thread_ts}",
        limit=1,
        token=token,
    )

    message = result["messages"][0]
    return message


def get_adcopy_blocks() -> List[Dict[str, Any]]:
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
