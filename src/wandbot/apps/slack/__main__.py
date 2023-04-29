import logging
from functools import partial

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from wandbot.api.client import APIClient
from wandbot.api.schemas import APIQueryResponse
from wandbot.apps.slack.config import SlackAppConfig
from wandbot.database.schemas import QuestionAnswer

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

config = SlackAppConfig()
app = App(token=config.SLACK_APP_TOKEN)
api_client = APIClient(url=config.WANDBOT_API_URL)


def format_response(response: APIQueryResponse | None, outro_message: str = "") -> str:
    if response is not None:
        result = response.answer
        if response.model != "gpt-4":
            warning_message = f"""**Warning: Falling back to {response.model}**, These results may nor be as good as **gpt-4**\n\n"""
            result = warning_message + response.answer

        if config.include_sources and response.sources:
            result = f"{result}\n\n**References**\n\n" + "- ".join(
                response.sources.splitlines()
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

        chat_history = api_client.get_chat_history(thread_id)

        if not chat_history:
            # send out the intro message
            send_message(
                say=say,
                message=f"Hi <@{user}>:\n\n{config.INTRO_MESSAGE}",
                thread=thread_id,
            )
        # process the query through the api
        api_response = api_client.query(query, thread_id, chat_history=chat_history)
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
        api_client.save_chat_history(
            [
                QuestionAnswer(
                    **api_response.dict(), question_answer_id=sent_message["ts"]
                )
            ]
        )

    except Exception as e:
        logger.error(f"Error posting message: {e}")


@app.event("reaction_added")
def handle_reaction_added(event, say):
    print(event)
    channel_id = event["item"]["channel"]
    message_ts = event["item"]["ts"]
    result = app.client.conversations_history(
        channel=channel_id,
        latest=message_ts,
        limit=-1,
        inclusive=True,
        token=config.SLACK_BOT_TOKEN,
    )
    print(result)

    # TODO: Add feedback handling

    # if result["ok"] and len(result["messages"]) > 0:
    #     # TODO: More robust way to handle feedback. This will base it on the first message in the thread
    #     # BUG: Message also returns the user info so stripping that before update
    #     query = result["messages"][0]["text"]
    #     if query is not None and isinstance(query, str):
    #         query = remove_angle_brackets_and_whitespace(query)
    #     user = event["user"]
    #     feedback = None
    #     print(str(event["reaction"]))
    #     if "+1" in str(event["reaction"]):
    #         feedback = "positive"
    #     if "-1" in str(event["reaction"]):
    #         feedback = "negative"
    #     if feedback:
    #         print("added feedback")
    #         slack_adapter.update_feedback_in_db(user, query, feedback)
    #     else:
    #         print(f"Unhandled reaction: {event['reaction']} in channel {channel_id}")
    # else:
    #     print(
    #         f"Unable to retrieve message for reaction {event['reaction']} in channel {channel_id}"
    #     )


if __name__ == "__main__":
    handler = SocketModeHandler(app, config.SLACK_APP_TOKEN)
    handler.start()
