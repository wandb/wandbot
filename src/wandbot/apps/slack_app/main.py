import os

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from wandbot.apps.chat import Chat

app = App(token=os.environ.get("SLACK_APP_TOKEN"))

chat = Chat()


@app.event("app_mention")
def command_handler(body, say, logger):
    try:
        text = body["event"].get("text")
        user = body["event"].get("user")
        thread_ts = body["event"].get("thread_ts", None) or body["event"].get(
            "ts", None
        )
        text = " ".join(text.split()[1:])
        response = chat(text)
        response = f"Hi <@{user}>:\n\n{response}"
        if thread_ts is not None:
            say(
                text=response,
                token=os.environ.get("SLACK_BOT_TOKEN"),
                thread_ts=thread_ts,
            )
        else:
            say(
                text=response,
                token=os.environ.get("SLACK_BOT_TOKEN"),
            )
    except Exception as e:
        logger.error(f"Error posting message: {e}")


@app.event("reaction_added")
def handle_reaction_added(event, say):
    channel_id = event["item"]["channel"]
    message_ts = event["item"]["ts"]
    result = app.client.conversations_history(
        channel=channel_id, latest=message_ts, limit=1, inclusive=True
    )

    if result["ok"] and len(result["messages"]) > 0:
        message = result["messages"][0]
        print(f"A user reacted with {event['reaction']} to message: {message['text']}")
    else:
        print(
            f"Unable to retrieve message for reaction {event['reaction']} in channel {channel_id}"
        )


if __name__ == "__main__":
    handler = SocketModeHandler(app, os.environ.get("SLACK_APP_TOKEN"))
    handler.start()
