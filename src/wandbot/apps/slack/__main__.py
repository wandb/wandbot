from collections import OrderedDict
from functools import partial

import langdetect
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from wandbot.api.client import APIClient
from wandbot.api.schemas import APIQueryResponse
from wandbot.apps.slack.config import SlackAppConfig
from wandbot.utils import get_logger

logger = get_logger(__name__)

config = SlackAppConfig()
app = App(token=config.SLACK_APP_TOKEN)
api_client = APIClient(url=config.WANDBOT_API_URL)


def deduplicate(input_list):
    return list(OrderedDict.fromkeys(input_list))


def format_response(response: APIQueryResponse | None, outro_message: str = "", lang: str = "en") -> str:
    if response is not None:
        result = response.answer
        if "gpt-4" not in response.model:
            if lang == "ja":
                warning_message = f"*警告: {response.model}* にフォールバックします。これらの結果は *gpt-4* ほど良くない可能性があります*"
            else:
                warning_message = (
                    f"*Warning: Falling back to {response.model}*, These results may nor be as good as " f"*gpt-4*\n\n"
                )
            result = warning_message + response.answer

        if config.include_sources and response.sources:
            sources_list = deduplicate(
                [item for item in response.sources.split(",") if item.strip().startswith("http")]
            )
            if len(sources_list) > 0:
                items = min(len(sources_list), 3)
                if lang == "ja":
                    result = f"{result}\n\n*参考文献*\n\n>" + "\n> ".join(sources_list[:items]) + "\n\n"
                else:
                    result = f"{result}\n\n*References*\n\n>" + "\n> ".join(sources_list[:items]) + "\n\n"
        if outro_message:
            result = f"{result}\n\n{outro_message}"

    else:
        if lang == "ja":
            result = config.JA_ERROR_MESSAGE
        else:
            result = config.EN_ERROR_MESSAGE
    return result


def send_message(say, message, thread=None):
    if thread is not None:
        return say(text=message, thread_ts=thread)
    else:
        return say(text=message)


@app.event("app_mention")
def command_handler(body, say, logger):
    try:
        query = body["event"].get("text")
        lang_code = langdetect.detect(query)
        user = body["event"].get("user")
        thread_id = body["event"].get("thread_ts", None) or body["event"].get("ts", None)
        say = partial(say, token=config.SLACK_BOT_TOKEN)

        chat_history = api_client.get_chat_history(application=config.APPLICATION, thread_id=thread_id)

        if not chat_history:
            # send out the intro message
            if lang_code == "ja":
                send_message(
                    say=say,
                    message=f"こんにちは <@{user}>:\n\n{config.JA_INTRO_MESSAGE}",
                    thread=thread_id,
                )
            else:
                send_message(
                    say=say,
                    message=f"Hi <@{user}>:\n\n{config.EN_INTRO_MESSAGE}",
                    thread=thread_id,
                )
        # process the query through the api
        api_response = api_client.query(question=query, chat_history=chat_history)
        if lang_code == "ja":
            response = format_response(api_response, config.JA_OUTRO_MESSAGE, lang_code)
        else:
            response = format_response(api_response, config.EN_OUTRO_MESSAGE)

        # send the response
        sent_message = send_message(say=say, message=response, thread=thread_id)

        app.client.reactions_add(
            channel=body["event"]["channel"],
            timestamp=sent_message["ts"],
            name="thumbsup",
            token=config.SLACK_BOT_TOKEN,
        )
        app.client.reactions_add(
            channel=body["event"]["channel"],
            timestamp=sent_message["ts"],
            name="thumbsdown",
            token=config.SLACK_BOT_TOKEN,
        )

        #  save the question answer to the database
        api_client.create_question_answer(
            thread_id=thread_id,
            question_answer_id=sent_message["ts"],
            **api_response.model_dump(),
        )

    except Exception as e:
        logger.error(f"Error posting message: {e}")


def parse_reaction(reaction: str):
    if reaction == "+1":
        return 1
    elif reaction == "-1":
        return -1
    else:
        return 0


@app.event("reaction_added")
def handle_reaction_added(event, say):
    channel_id = event["item"]["channel"]
    message_ts = event["item"]["ts"]

    conversation = app.client.conversations_replies(
        channel=channel_id,
        ts=message_ts,
        inclusive=True,
        limit=1,
        token=config.SLACK_BOT_TOKEN,
    )
    messages = conversation.get(
        "messages",
    )
    if messages and len(messages):
        thread_ts = messages[0].get("thread_ts")
        if thread_ts:
            rating = parse_reaction(event["reaction"])
            api_client.create_feedback(
                feedback_id=event["event_ts"],
                question_answer_id=message_ts,
                rating=rating,
            )


if __name__ == "__main__":
    handler = SocketModeHandler(app, config.SLACK_APP_TOKEN)
    handler.start()
