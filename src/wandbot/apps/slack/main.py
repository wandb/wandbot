import os
from functools import partial

import requests
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from wandbot.apps.slack.config import SlackAppConfig

config = SlackAppConfig()
app = App(token=config.SLACK_APP_TOKEN)


def format_response(user, response):
    if response is not None:
        if config.include_sources:
            result = (
                response["answer"]
                + "\n\n**References**\n\n"
                + "- ".join(response["sources"].splitlines())
            )
        else:
            result = response["answer"]

        output = f"Hi <@{user}>:\n\n{result}"
    else:
        output = f"Hi <@{user}>:\n\n {config.ERROR_MESSAGE}"
    return output


def run_api_query(
    query: str,
    thread_id: str,
):
    try:
        query = {"question": query, "thread_id": thread_id, "application": "slack"}
        query_url = f"{config.WANDBOT_API_URL}/query"

        response = requests.post(query_url, json=query)
        if response.status_code == 200:
            response = response.json()
            return response
    except:
        return None


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
        thread_ts = body["event"].get("thread_ts", None) or body["event"].get(
            "ts", None
        )
        say = (partial(say, token=config.SLACK_BOT_TOKEN),)
        # send out the intro message
        send_message(
            say=say,
            message=f"Hi <@{user}>:\n\n{config.INTRO_MESSAGE}",
            thread=thread_ts,
        )

        # process the query through the api
        api_response = run_api_query(query, thread_ts)
        response = format_response(user, api_response)

        # send the response
        send_message(say=say, message=response, thread=thread_ts)

        # send the outro message
        outro_sent = send_message(
            say=say, message=config.OUTRO_MESSAGE, thread=thread_ts
        )

        app.client.reactions_add(
            channel=body["event"]["channel"],
            timestamp=outro_sent["ts"],
            name="thumbsup",
            token=config.SLACK_BOT_TOKEN,
        )
        app.client.reactions_add(
            channel=body["event"]["channel"],
            timestamp=outro_sent["ts"],
            name="thumbsdown",
            token=config.SLACK_BOT_TOKEN,
        )
    except Exception as e:
        logger.error(f"Error posting message: {e}")


if __name__ == "__main__":
    handler = SocketModeHandler(app, os.environ.get("SLACK_APP_TOKEN"))
    handler.start()
