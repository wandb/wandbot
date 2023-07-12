import logging
from functools import partial

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from wandbot.api.client import APIClient
from wandbot.api.schemas import APIQueryResponse
from wandbot.apps.slack.config import SlackAppConfig

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

config = SlackAppConfig()
app = App(token=config.SLACK_APP_TOKEN)
api_client = APIClient(url=config.WANDBOT_API_URL)


def format_response(response: APIQueryResponse | None, outro_message: str = "") -> str:
    if response is not None:
        result = response.answer
        if response.model != "gpt-4":
            warning_message = (
                f"*Warning: Falling back to {response.model}*, These results may nor be as good as "
                f"*gpt-4*\n\n"
            )
            result = warning_message + response.answer

        if config.include_sources and response.sources:
            sources_list = [
                item
                for item in response.sources.split(",")
                if item.strip().startswith("http")
            ]
            if len(sources_list) > 0:
                result = (
                    f"{result}\n\n*References*\n\n>"
                    + "\n> ".join(sources_list)
                    + "\n\n"
                )
        if outro_message:
            result = f"{result}\n\n{outro_message}"

    else:
        result = config.ERROR_MESSAGE
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
        user = body["event"].get("user")
        thread_id = body["event"].get("thread_ts", None) or body["event"].get(
            "ts", None
        )
        say = partial(say, token=config.SLACK_BOT_TOKEN)

        chat_history = api_client.get_chat_history(
            application=config.APPLICATION, thread_id=thread_id
        )

        if not chat_history:
            # send out the intro message
            send_message(
                say=say,
                message=f"Hi <@{user}>:\n\n{config.INTRO_MESSAGE}",
                thread=thread_id,
            )
        # process the query through the api
        api_response = api_client.query(question=query, chat_history=chat_history)
        response = format_response(api_response, config.OUTRO_MESSAGE)

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
            **api_response.dict(),
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
