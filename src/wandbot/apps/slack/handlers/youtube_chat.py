import json
import logging

from pytube.exceptions import RegexMatchError
from slack_sdk.web.async_client import AsyncWebClient

from wandbot.apps.slack.utils import get_initial_message_from_thread
from wandbot.youtube_chat.video_utils import YoutubeVideoInfo


def get_youtube_chat_init_block():
    blocks = [
        {
            "type": "rich_text",
            "elements": [
                {
                    "type": "rich_text_section",
                    "elements": [
                        {
                            "type": "text",
                            "text": "Enter the URL of the youtube video\n",
                        },
                        {
                            "type": "text",
                            "text": "Note: Only supports the following URL patterns\n\n",
                            "style": {"bold": True},
                        },
                    ],
                },
                {
                    "type": "rich_text_list",
                    "style": "bullet",
                    "elements": [
                        {
                            "type": "rich_text_section",
                            "elements": [
                                {
                                    "type": "text",
                                    "text": "https://youtube.com/watch?v={video_id}",
                                    "style": {"italic": True},
                                }
                            ],
                        },
                        {
                            "type": "rich_text_section",
                            "elements": [
                                {
                                    "type": "text",
                                    "text": "https://youtube.com/embed/{video_id}",
                                    "style": {"italic": True},
                                }
                            ],
                        },
                        {
                            "type": "rich_text_section",
                            "elements": [
                                {
                                    "type": "text",
                                    "text": "https://youtu.be/{video_id}",
                                    "style": {"italic": True},
                                }
                            ],
                        },
                    ],
                },
            ],
        },
        {"type": "divider"},
        {
            "type": "input",
            "element": {
                "type": "url_text_input",
                "action_id": "chat_youtube",
                "dispatch_action_config": {
                    "trigger_actions_on": ["on_enter_pressed"]
                },
                "focus_on_load": True,
                "placeholder": {
                    "type": "plain_text",
                    "text": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                },
            },
            "label": {
                "type": "plain_text",
                "text": "Youtube URL",
                "emoji": True,
            },
        },
    ]
    return blocks


def get_video_confirmation_blocks(url):
    try:
        block_json = """[
            {{
                "type": "section",
                "text": {{
                    "type": "plain_text",
                    "text": "Making sure I have the correct video"
                }}
            }},
            {{
                "type": "video",
                "title": {{
                    "type": "plain_text",
                    "text": "{title}",
                    "emoji": true
                }},
                "title_url": "{video_url}",
                "description": {{
                    "type": "plain_text",
                    "text": "{description}",
                    "emoji": true
                }},
                "video_url": "{embed_url}",
                "alt_text": "How to use Slack?",
                "thumbnail_url": "{thumbnail_url}",
                "author_name": "{author}",
                "provider_name": "YouTube",
                "provider_icon_url": "https://a.slack-edge.com/80588/img/unfurl_icons/youtube.png"
            }}, 
            {{"type": "divider"}},
            {{
                "type": "actions",
                "elements": [
                    {{
                        "type": "button",
                        "text": {{
                            "type": "plain_text",
                            "emoji": true,
                            "text": "Approve"
                        }},
                        "style": "primary",
                        "value": "yt_approve",
                        "action_id": "yt_approve"
                    }},
                    {{
                        "type": "button",
                        "text": {{
                            "type": "plain_text",
                            "emoji": true,
                            "text": "Deny"
                        }},
                        "style": "danger",
                        "value": "yt_deny",
                        "action_id": "yt_deny"
                    }}
                ]
            }}
        ]"""
        video_info = YoutubeVideoInfo(url=url)
        filled_block = block_json.format(
            title=video_info.title,
            video_url=url,
            description=video_info.description,
            embed_url=video_info.embed_url,
            thumbnail_url=video_info.thumbnail_url,
            author=video_info.author,
        )

        blocks = json.loads(filled_block)
    except RegexMatchError as e:
        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Sorry!*\n{url}\n *doesn't look to be a valid YouTube URL.*\n*Please try again with a "
                    f"valid URL *",
                },
            }
        ]
    return blocks


def create_youtube_chat_init_handler(slack_client: AsyncWebClient) -> callable:
    async def youtube_chat_init_handler(
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
        blocks = get_youtube_chat_init_block()
        await say(blocks=blocks, thread_ts=thread_ts)

    return youtube_chat_init_handler


def create_youtube_chat_input_handler() -> callable:
    async def handle_youtube_chat_input(
        ack: callable, body: dict, say: callable, logger: logging.Logger
    ) -> None:
        await ack()
        logger.info(f"Received message: {body}")
        url = body["actions"][0]["value"]
        await say(
            "Working on in it...",
            thread_ts=body["message"]["thread_ts"],
        )
        video_confirmation_block = get_video_confirmation_blocks(url)
        await say(
            blocks=video_confirmation_block,
            thread_ts=body["message"]["thread_ts"],
        )

    return handle_youtube_chat_input


def create_youtube_approve_handler() -> callable:
    async def handle_youtube_approve(
        ack: callable, body: dict, say: callable, logger: logging.Logger
    ) -> None:
        await ack()
        logger.info(f"Received message: {body}")
        # TODO: Handle the approved youtube video
        # Steps: Fetch Transcript and Create an assistant for the video
